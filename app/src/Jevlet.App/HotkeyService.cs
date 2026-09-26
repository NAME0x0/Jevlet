using System.Runtime.InteropServices;
using System.Windows.Interop;

namespace Jevlet.App;

/// <summary>
/// The palette hotkey, delivered two independent ways: RegisterHotKey on a message-only window,
/// and a low-level keyboard hook on its own thread. The hook catches the chord even when another
/// program (a launcher, an AutoHotkey script) intercepts it before RegisterHotKey fires; when the
/// chord could not be registered, the hook also swallows it so Windows does not act on it.
/// Presses within 250 ms of each other count once, and every delivery is logged.
/// </summary>
internal sealed class HotkeyService : IDisposable
{
    private const int HotkeyId = 0x4A45; // "JE"
    private static readonly (string Name, uint Modifiers, uint Key)[] Fallbacks =
    [
        ("Alt+Shift+Space", Native.MOD_ALT | Native.MOD_SHIFT, 0x20),
        ("Ctrl+Alt+J", Native.MOD_CONTROL | Native.MOD_ALT, 0x4A),
    ];

    private readonly HwndSource _window;
    private readonly Action _pressed;
    private readonly Thread _hookThread;
    private uint _hookThreadId;
    private nint _hook;
    private Native.LowLevelKeyboardProc? _hookCallback; // kept alive for the native hook
    private long _lastPress;
    private bool _chordHeld;

    public string Active { get; private set; } = "";
    public uint Modifiers { get; private set; }
    public uint Key { get; private set; }
    public bool Registered { get; private set; }

    public HotkeyService(string chord, Action pressed)
    {
        _pressed = pressed;
        // A message-only window: never shown, never in Alt-Tab, alive as long as the app.
        _window = new HwndSource(new HwndSourceParameters("JevletHotkey") { ParentWindow = new nint(-3), WindowStyle = 0 });
        _window.AddHook(WndProc);
        Register(chord);
        _hookThread = new Thread(HookLoop) { IsBackground = true, Name = "Jevlet keyboard hook" };
        _hookThread.Start();
    }

    private void Register(string chord)
    {
        var candidates = new List<(string, uint, uint)>();
        if (TryParse(chord, out var modifiers, out var key))
        {
            candidates.Add((chord, modifiers, key));
        }
        candidates.AddRange(Fallbacks.Where(f => f.Name != chord));
        foreach (var (name, mods, vk) in candidates)
        {
            if (Native.RegisterHotKey(_window.Handle, HotkeyId, mods | Native.MOD_NOREPEAT, vk))
            {
                (Active, Modifiers, Key, Registered) = (name, mods, vk, true);
                Log.Info($"hotkey registered: {name}");
                return;
            }
            Log.Info($"hotkey {name} unavailable (error {Marshal.GetLastWin32Error()}); another program holds it");
        }
        // Nothing registered: the keyboard hook alone serves the requested chord.
        (Active, Modifiers, Key, Registered) = (chord, modifiers, key, false);
        Log.Info($"hotkey {chord} served by the keyboard hook only");
    }

    public static bool TryParse(string chord, out uint modifiers, out uint key)
    {
        modifiers = 0;
        key = 0;
        foreach (var part in chord.Split('+', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            switch (part.ToUpperInvariant())
            {
                case "ALT": modifiers |= Native.MOD_ALT; break;
                case "CTRL" or "CONTROL": modifiers |= Native.MOD_CONTROL; break;
                case "SHIFT": modifiers |= Native.MOD_SHIFT; break;
                case "WIN": modifiers |= Native.MOD_WIN; break;
                case "SPACE": key = 0x20; break;
                case var letter when letter.Length == 1 && char.IsAsciiLetterOrDigit(letter[0]): key = letter[0]; break;
                case var function when function.StartsWith('F') && int.TryParse(function[1..], out var n) && n is >= 1 and <= 24:
                    key = (uint)(0x70 + n - 1);
                    break;
                default: return false;
            }
        }
        return key != 0 && modifiers != 0;
    }

    private nint WndProc(nint hwnd, int message, nint wParam, nint lParam, ref bool handled)
    {
        if (message == Native.WM_HOTKEY && wParam == HotkeyId)
        {
            handled = true;
            Deliver("RegisterHotKey");
        }
        return 0;
    }

    private void Deliver(string source)
    {
        var now = Environment.TickCount64;
        if (now - Interlocked.Exchange(ref _lastPress, now) < 250)
        {
            return; // the other path already delivered this press
        }
        Log.Info($"hotkey pressed ({source})");
        _window.Dispatcher.BeginInvoke(_pressed);
    }

    private void HookLoop()
    {
        _hookThreadId = Native.GetCurrentThreadId();
        _hookCallback = HookCallback;
        _hook = Native.SetWindowsHookEx(Native.WH_KEYBOARD_LL, _hookCallback, Native.GetModuleHandle(null), 0);
        if (_hook == 0)
        {
            Log.Error($"keyboard hook unavailable (error {Marshal.GetLastWin32Error()})");
            return;
        }
        while (Native.GetMessage(out var message, 0, 0, 0) > 0)
        {
            // Low-level hooks need a message loop on their thread; nothing else happens here.
            _ = message;
        }
        Native.UnhookWindowsHookEx(_hook);
    }

    private nint HookCallback(int code, nint wParam, nint lParam)
    {
        if (code >= 0)
        {
            var info = Marshal.PtrToStructure<Native.KBDLLHOOKSTRUCT>(lParam);
            if (info.vkCode == Key && (info.flags & Native.LLKHF_INJECTED) == 0)
            {
                var down = wParam == Native.WM_KEYDOWN || wParam == Native.WM_SYSKEYDOWN;
                if (!down)
                {
                    _chordHeld = false;
                }
                else if (ModifiersDown())
                {
                    if (!_chordHeld)
                    {
                        _chordHeld = true; // key repeat while held must not reopen the palette
                        Deliver("keyboard hook");
                    }
                    if (!Registered)
                    {
                        return 1; // we own this chord; keep Windows from acting on it
                    }
                }
            }
        }
        return Native.CallNextHookEx(_hook, code, wParam, lParam);
    }

    private bool ModifiersDown()
    {
        static bool Down(int key) => (Native.GetAsyncKeyState(key) & 0x8000) != 0;
        bool Want(uint flag) => (Modifiers & flag) != 0;
        return Want(Native.MOD_ALT) == Down(0x12)
            && Want(Native.MOD_CONTROL) == Down(0x11)
            && Want(Native.MOD_SHIFT) == Down(0x10)
            && Want(Native.MOD_WIN) == (Down(0x5B) || Down(0x5C));
    }

    public void Dispose()
    {
        Native.UnregisterHotKey(_window.Handle, HotkeyId);
        if (_hookThreadId != 0)
        {
            Native.PostThreadMessage(_hookThreadId, 0x0012 /* WM_QUIT */, 0, 0);
        }
        _window.Dispose();
    }
}
