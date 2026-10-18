# SPDX-License-Identifier: GPL-2.0-or-later
"""The resource system's lock claim backs off from an exclusive pass (prepare/resource_lock.py)."""
import ctypes
import hashlib
import os
import struct

import pytest

import dcport.prepare_state  # noqa: F401  (puts prepare/ on sys.path)
import resource_lock

A, M = 0x7592040, 0x7591620  # two scePthreadSelf values: a worker and the exclusive pass's thread
RWLOCK = 0x2000              # the +0x200 lock handle, passed to pthread_rwlock_unlock
STEP1, RESUME = 1, 2


@pytest.fixture(scope="module")
def run_claim():
    """Runs the trampoline natively against a fake lock object: (fields, self) -> (exit, fields,
    unlock calls, unlock argument). fields: state, count, owner, exclusive thread."""
    if os.name != "nt" or struct.calcsize("P") != 8:
        pytest.skip("runs x86-64 code with the Windows x64 calling convention")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
    k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    base = k32.VirtualAlloc(None, 0x1000, 0x3000, 0x40)
    assert base
    # rcx = lock object, rdx = self, r8 = result. Sets up the routine's registers at the claim
    # (r13 = lock, rbx = lock+0x100, r12 = lock+0x104) and jumps to the trampoline at +0x100;
    # scePthreadSelf, pthread_rwlock_unlock, step 1 and the resume point are stubs here.
    harness = bytearray(bytes.fromhex("57" "53" "4154" "4155" "4156" "4157"   # save
                                      "4989cd" "498d9d00010000" "4d8da504010000"
                                      "4989d6" "4d89c7" "e9"))
    harness += struct.pack("<i", 0x100 - (len(harness) + 4))
    self_fn = len(harness)
    harness += bytes.fromhex("4c89f0" "c3")                         # rax = self
    unlock_fn = len(harness)
    harness += bytes.fromhex("49ff07" "49897f10" "c3")              # count, record rdi
    step1 = len(harness)
    harness += bytes.fromhex("49c7470801000000" "eb08")
    resume = len(harness)
    harness += bytes.fromhex("49c7470802000000")
    harness += bytes.fromhex("415f" "415e" "415d" "415c" "5b" "5f" "c3")  # restore
    assert len(harness) < 0x100
    code = harness.ljust(0x100, b"\xcc") + resource_lock.trampoline(
        base + 0x100, base + self_fn, base + unlock_fn, base + step1, base + resume)
    ctypes.memmove(base, bytes(code), len(code))
    function = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)(base)

    def run(state, count, owner, exclusive, self):
        lock = ctypes.create_string_buffer(0x210)
        struct.pack_into("<IIQQ", lock, 0x100, state, count, owner, exclusive)
        struct.pack_into("<Q", lock, 0x200, RWLOCK)
        result = (ctypes.c_uint64 * 3)()
        function(ctypes.addressof(lock), self, ctypes.addressof(result))
        return result[1], struct.unpack_from("<IIQQ", lock.raw, 0x100), result[0], result[2]
    yield run
    k32.VirtualFree(base, 0, 0x8000)


@pytest.mark.parametrize("state", [2, 3])
def test_a_registered_claim_runs_the_original_code(run_claim, state):
    assert run_claim(state, 1, 0, 0, A) == (RESUME, (state, 1, A, 0), 0, 0)


def test_the_exclusive_pass_claims_as_before(run_claim):
    assert run_claim(4, 1, 0, M, M) == (RESUME, (4, 1, M, M), 0, 0)


def test_a_claim_inside_the_exclusive_pass_backs_off(run_claim):
    # straight from step 3: its count goes back, the owner field stays empty
    assert run_claim(4, 1, 0, M, A) == (STEP1, (4, 0, 0, M), 1, RWLOCK)
    # after waiting in step 2, which set owner=self: the pass waits for owner 0, so clear it
    assert run_claim(4, 1, A, M, A) == (STEP1, (4, 0, 0, M), 1, RWLOCK)
    # the pass started but has not set +0x110 yet
    assert run_claim(4, 1, A, 0, A) == (STEP1, (4, 0, 0, 0), 1, RWLOCK)


def test_backing_off_leaves_another_owner_alone(run_claim):
    assert run_claim(4, 2, M, M, A) == (STEP1, (4, 1, M, M), 1, RWLOCK)


def test_a_claim_the_ended_pass_left_unregistered_starts_over(run_claim):
    # both state changes failed while the pass ran; it has ended since (4->0, then 0->1)
    assert run_claim(1, 1, 0, 0, A) == (STEP1, (1, 0, 0, 0), 1, RWLOCK)


def test_trampoline_instructions():
    capstone = pytest.importorskip("capstone")
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    code = resource_lock.trampoline(0x2b30000, 0x7ae780, 0x12402d0, 0x77e7dc, 0x77e8ce)
    listing = [(i.address, i.mnemonic, i.op_str) for i in md.disasm(code, 0x2b30000)]
    assert sum(len(i.bytes) for i in md.disasm(code, 0x2b30000)) == len(code)
    assert [m for _, m, _ in listing] == [
        "mov", "sub", "cmp", "jbe", "call", "cmp", "je", "lock dec", "cmp", "jne", "mov",
        "mov", "call", "jmp", "call", "mov", "jmp"]
    ops = {m + " " + o for _, m, o in listing}
    assert {"call 0x7ae780", "call 0x12402d0", "jmp 0x77e7dc", "jmp 0x77e8ce",
            "lock dec dword ptr [r12]"} <= ops
    assert listing[3][2] == hex(listing[14][0]) and listing[6][2] == hex(listing[15][0])


# A routine shaped like 01.28's around the claim, in a small image.
ROUTINE = dict(start=0x10, end=0x80, site=0x40, step1=0x20, resume=0x4c, unlock_call=0x50)


def small_image():
    image = bytearray(b"\x90" * 0x200)
    image[0x40:0x45] = b"\xe8" + struct.pack("<i", 0x100 - 0x45)      # call scePthreadSelf
    image[0x45:0x4c] = resource_lock.CLAIM
    image[0x50:0x55] = b"\xe8" + struct.pack("<i", 0x180 - 0x55)      # call pthread_rwlock_unlock
    routine = dict(ROUTINE, sha256=hashlib.sha256(image[0x10:0x80]).hexdigest())
    return image, routine


def test_patch_redirects_the_claim_into_a_new_segment():
    image, routine = small_image()
    patched, segments, report = resource_lock.patch(bytes(image), [(0, 0x200, 5)], 0x200,
                                                    active=True, routine=routine)
    assert report["applied"] and report["trampoline"] == "0x10000"
    assert segments == [(0, 0x200, 5), (0x10000, 0x1000, 5)]
    assert patched[0x40] == 0xe9 and 0x45 + struct.unpack_from("<i", patched, 0x41)[0] == 0x10000
    assert patched[0x45:0x4c] == b"\x90" * 7
    assert patched[:0x40] == image[:0x40] and patched[0x4c:0x200] == image[0x4c:]
    body = resource_lock.trampoline(0x10000, 0x100, 0x180, 0x20, 0x4c)
    assert patched[0x10000:0x10000 + len(body)] == body
    assert len(patched) == 0x11000


@pytest.mark.parametrize("change", ["version", "inactive", "community patch", "relocation", "segment"])
def test_patch_leaves_the_image_alone(change):
    image, routine = small_image()
    args = dict(segments=[(0, 0x200, 5)], relocations=[], patched=[], active=True)
    if change == "version":
        image[0x30] = 0xcc
    elif change == "inactive":
        args["active"] = False
    elif change == "community patch":
        args["patched"] = [0x60]
    elif change == "relocation":
        args["relocations"] = [(0x44, 0, 0x1000, 0)]
    else:
        args["segments"] = [(0, 0x200, 6)]
    patched, segments, report = resource_lock.patch(
        bytes(image), args["segments"], 0x200, args["relocations"], args["patched"],
        active=args["active"], routine=routine)
    assert patched == image and segments == args["segments"]
    assert not report["applied"] and report["reason"]


def test_dc_lock_patch_switch(monkeypatch):
    monkeypatch.delenv("DC_LOCK_PATCH", raising=False)
    assert resource_lock.enabled()
    monkeypatch.setenv("DC_LOCK_PATCH", "0")
    assert not resource_lock.enabled()
