# SPDX-License-Identifier: GPL-2.0-or-later
"""DRIVECLUB 01.28: the resource system's lock froze the game in car selection after a race.

The lock (state at +0x100: 0 idle, 1 pending, 2 owned, 3 owned+pending, 4 exclusive; holder
count +0x104, owner +0x108, thread of the exclusive pass +0x110) is taken by one routine
(0x77e7b0), which also does the work and releases it: (1) while the state is 4, wait unless
this thread runs the exclusive pass; (2) wait for another owner to let go; (3) count+1 and, for
the first holder, state 0->2 or 1->3 and owner=self. The exclusive pass starts with state
1->4 under a lock the routine never takes, so it can land between a thread's step 1 and its
step 3. That thread's state change then fails, yet it still becomes owner and runs its work
during the pass. When that work takes the lock again (a resource it releases unloads another),
it waits in step 1 for the pass to end while the pass waits in step 2 for it to let go, and
the game freezes (main in the pass, async_1 the owner). Step 2 waits with sceKernelUsleep(1),
so the window lasts as long as the previous owner holds the lock.

The claim (`call scePthreadSelf; mov [lock+0x108], rax`) jumps to a trampoline in a new
executable segment. When the state change did not register (the state is not 2 or 3) and this
thread does not run the exclusive pass, it gives its count back, clears the owner field if it
had set it, unlocks and starts over at step 1, which waits for the pass to end. Every other
path runs the original instructions. Other game versions are left alone (the routine's SHA-256
must match); DC_LOCK_PATCH=0 disables the fix.
"""
import bisect
import hashlib
import os
import struct

# Offsets in the 01.28 eboot. `site`: call scePthreadSelf (rel32) + mov [r13+0x108], rax, where
# rbx = lock+0x100 and r12 = lock+0x104; `step1`: the exclusive-mode check, which expects rbx
# as here; `resume`: the instruction after the claim; `unlock_call`: the routine's call to
# pthread_rwlock_unlock for the lock at +0x200 (held from step 2 to here).
ROUTINE = dict(start=0x77e7b0, end=0x77ebb0,
               sha256='e91a956850117f43d89970ce91eb376045561498da6782520c3abab891e67da2',
               site=0x77e8c2, step1=0x77e7dc, resume=0x77e8ce, unlock_call=0x77e8d9)
CLAIM = bytes.fromhex('49898508010000')  # mov [r13+0x108], rax
MAX_IMAGE_SIZE = 512 * 1024 * 1024


def enabled():
    return os.environ.get('DC_LOCK_PATCH') != '0'


def call_target(image, at):
    """Destination of the `call rel32` at `at`."""
    if image[at] != 0xe8:
        raise ValueError(f'no call at {at:#x}')
    return at + 5 + struct.unpack_from('<i', image, at + 1)[0]


def trampoline(at, self_fn, unlock_fn, step1, resume):
    """The claim with a back-off, placed at `at`; the other arguments are absolute targets."""
    def rel(target, end):
        return struct.pack('<i', target - (at + end))
    code = bytearray()
    code += bytes.fromhex('8b03' '83e802' '83f801' '7638')   # state = [rbx]; 2 or 3: claim
    code += b'\xe8' + rel(self_fn, len(code) + 5)            # rax = self
    code += bytes.fromhex('493b8510010000' '742f')           # the exclusive pass's thread: own
    code += bytes.fromhex('f041ff0c24')                      # lock dec dword [r12]: count back
    code += bytes.fromhex('493b8508010000' '750b')           # owner set by this thread (step 2):
    code += bytes.fromhex('49c7850801000000000000')          #   owner = 0
    code += bytes.fromhex('498bbd00020000')                  # rdi = [r13+0x200]
    code += b'\xe8' + rel(unlock_fn, len(code) + 5)          # pthread_rwlock_unlock
    code += b'\xe9' + rel(step1, len(code) + 5)              # start over at step 1
    assert len(code) == 66                                   # claim:
    code += b'\xe8' + rel(self_fn, len(code) + 5)            # rax = self
    assert len(code) == 71                                   # own:
    code += CLAIM                                            # owner = self
    code += b'\xe9' + rel(resume, len(code) + 5)
    return bytes(code)


def patch(image, segments, main_size, relocations=(), patched=(), *, active=None, routine=ROUTINE):
    """A copy of the linked image with the claim redirected, the segment list with the trampoline
    segment added, and a report. Unchanged when inactive or the routine is not 01.28's.
    patched: eboot offsets the community patches write."""
    is_active = enabled() if active is None else active
    report = dict(enabled=bool(is_active), applied=False, reason=None, trampoline=None)
    result, new_segments = bytearray(image), list(segments)
    if not is_active:
        report['reason'] = 'disabled (DC_LOCK_PATCH=0)'
        return result, new_segments, report
    start, end, site, resume = routine['start'], routine['end'], routine['site'], routine['resume']
    if end > main_size or hashlib.sha256(image[start:end]).hexdigest() != routine['sha256']:
        report['reason'] = 'routine not found (another game version)'
        return result, new_segments, report
    if not any(flags & 1 and base <= start and end <= base + size for base, size, flags in segments):
        report['reason'] = 'routine outside the executable segment'
        return result, new_segments, report
    if any(start <= offset < end for offset in patched):
        report['reason'] = 'a community patch writes the routine'
        return result, new_segments, report
    targets = sorted(r[0] for r in relocations)
    i = bisect.bisect_left(targets, site - 7)
    if i < len(targets) and targets[i] < resume:
        report['reason'] = 'the claim overlaps a relocation'
        return result, new_segments, report
    if image[site + 5:resume] != CLAIM:
        raise ValueError('resource lock: the claim does not match the routine')
    arena = (len(image) + 65535) & ~65535
    body = trampoline(arena, call_target(image, site), call_target(image, routine['unlock_call']),
                      routine['step1'], resume)
    padded = (arena + len(body) + 4095) & ~4095
    if padded > MAX_IMAGE_SIZE:
        raise ValueError('resource lock trampoline exceeds the image size limit')
    result.extend(bytes(arena - len(result)) + body + b'\xcc' * (padded - arena - len(body)))
    result[site:resume] = b'\xe9' + struct.pack('<i', arena - (site + 5)) + b'\x90' * (resume - site - 5)
    new_segments.append((arena, padded - arena, 5))
    report.update(applied=True, trampoline=hex(arena))
    return result, new_segments, report
