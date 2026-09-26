using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Management;
using System.Windows;
using Jevlet.Core;
using Microsoft.Win32;

namespace Jevlet.App;

/// <summary>What was on screen when the palette opened, and how to map chosen labels back.</summary>
internal sealed record CommandContext(nint Previous, IReadOnlyList<OpenWindow> Windows, IReadOnlyDictionary<string, long> StoreIds);

/// <summary>The result of running a plan. Lines keep the palette open to show an answer or a list.</summary>
internal sealed record Outcome(bool Ok, string Message, IReadOnlyList<string>? Lines = null)
{
    public bool KeepOpen => Lines is not null;
}

/// <summary>Carries out a plan. Every skill in shared/skills.json has a branch here.</summary>
internal sealed class Executor(Stores stores, Scheduler scheduler, Settings settings)
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public async Task<Outcome> Run(Plan plan, CommandContext context)
    {
        var args = plan.Arguments;
        string Arg(string slot) => args.TryGetValue(slot, out var value) ? value : "";
        var now = DateTime.Now;
        try
        {
            switch (plan.Skill.Key)
            {
                case "open_app":
                    var app = Desktop.Apps.FirstOrDefault(a => a.Name == Arg("app"));
                    if (app is null)
                    {
                        return new Outcome(false, $"{Arg("app")} is not installed");
                    }
                    Process.Start(new ProcessStartInfo("explorer.exe", $"shell:AppsFolder\\{app.AppId}") { UseShellExecute = true });
                    return new Outcome(true, $"Opening {app.Name}");
                case "switch_window" or "close_window" or "minimize_window" or "maximize_window":
                    return WindowAction(plan.Skill.Key, Arg("window"), context);
                case "media":
                    Native.SendKeys(Arg("media") switch { "Next track" => (ushort)0xB0, "Previous track" => (ushort)0xB1, _ => (ushort)0xB3 });
                    return new Outcome(true, Arg("media"));
                case "volume":
                    var key = Arg("volume") switch { "Volume up" => (ushort)0xAF, "Volume down" => (ushort)0xAE, _ => (ushort)0xAD };
                    for (var press = 0; press < (key == 0xAD ? 1 : 5); press++)
                    {
                        Native.SendKeys(key);
                    }
                    return new Outcome(true, Arg("volume"));
                case "brightness":
                    return Brightness(Arg("brightness") == "Brighter" ? 20 : -20);
                case "radio":
                    return await Radio(Arg("radio")).ConfigureAwait(true);
                case "settings" or "website" or "folder":
                    var slot = plan.Skill.Key == "settings" ? "setting" : plan.Skill.Key;
                    var target = Catalogue.Shared.Targets[slot].First(t => t.Key == Arg(slot));
                    Shell(target.Value);
                    return new Outcome(true, $"Opening {target.Key}");
                case "theme":
                    return Theme(Arg("theme") == "Dark mode");
                case "search":
                    Shell($"https://www.google.com/search?q={Uri.EscapeDataString(Arg("text"))}");
                    return new Outcome(true, $"Searching for “{Arg("text")}”");
                case "open_file":
                    var path = FileSearch.PathOf(Arg("file"));
                    if (path is null)
                    {
                        return new Outcome(false, $"Couldn't find {Arg("file")}");
                    }
                    Shell(path);
                    return new Outcome(true, $"Opening {Path.GetFileName(path)}");
                case "type":
                    Focus(context.Previous);
                    await Task.Delay(120).ConfigureAwait(true);
                    Native.SendText(Arg("text"));
                    return new Outcome(true, "Typed");
                case "shortcut":
                    return await Shortcut(Arg("shortcut"), context.Previous).ConfigureAwait(true);
                case "click":
                    return new Outcome(false, "Clicking on screen arrives in the next update");
                case "timer":
                    var timer = scheduler.StartTimer(TimeSpan.FromSeconds(plan.DurationSeconds ?? 60));
                    return new Outcome(true, $"{timer.Label} started");
                case "timer_control":
                    return new Outcome(true, scheduler.ControlTimer(Arg("timer_action"), TextRules.ParseDuration(plan.Command)));
                case "stopwatch":
                    return new Outcome(true, scheduler.ControlStopwatch(Arg("stopwatch")));
                case "set_alarm":
                    var alarm = stores.AddAlarm(plan.When!.When.TimeOfDay, Arg("text"));
                    return new Outcome(true, $"Alarm set for {plan.When.When.ToString("h:mm tt", Invariant)}{(alarm.Label.Length > 0 ? $" ({alarm.Label})" : "")}");
                case "list_alarms":
                    var alarms = stores.Alarms();
                    return Lines(alarms.Count == 0 ? "No alarms" : $"{alarms.Count} alarm{Plural(alarms.Count)}",
                        alarms.Select(a => $"{DateTime.Today.Add(a.Time).ToString("h:mm tt", Invariant)}  {a.Label}"));
                case "cancel_alarm":
                    return Delete("alarm", Arg("alarm"), context, stores.CancelAlarm, "Alarm deleted");
                case "alarm_control":
                    return new Outcome(true, Arg("alarm_action") == "Snooze" ? scheduler.Snooze() : scheduler.StopRinging());
                case "set_reminder":
                    stores.AddReminder(Arg("text"), plan.When!.When);
                    return new Outcome(true, $"I'll remind you {plan.When.Describe(now)}");
                case "list_reminders":
                    var reminders = stores.PendingReminders()
                        .Where(r => plan.When is null || r.Due.Date == plan.When.When.Date).ToList();
                    return Lines(reminders.Count == 0 ? "No reminders" : $"{reminders.Count} reminder{Plural(reminders.Count)}",
                        reminders.Select(r => $"{new TimeMatch(r.Due, true, true).Describe(now)}  {r.Task}"));
                case "cancel_reminder":
                    return Delete("reminder", Arg("reminder"), context, stores.CancelReminder, "Reminder deleted");
                case "create_event":
                    stores.AddEvent(Arg("text"), plan.When!.When);
                    return new Outcome(true, $"Added “{Arg("text")}” {plan.When.Describe(now)}");
                case "show_agenda":
                    return Agenda(plan.When?.When ?? now, now);
                case "move_event":
                    if (!context.StoreIds.TryGetValue($"event:{Arg("event")}", out var eventId) || plan.When is null)
                    {
                        return new Outcome(false, "Which event, and when?");
                    }
                    stores.MoveEvent(eventId, plan.When.When);
                    return new Outcome(true, $"Moved to {plan.When.Describe(now)}");
                case "cancel_event":
                    return Delete("event", Arg("event"), context, stores.CancelEvent, "Event cancelled");
                case "add_todo":
                    stores.AddTodo(Arg("text"));
                    return new Outcome(true, $"Added to your list: {Arg("text")}");
                case "show_todos":
                    var todos = stores.OpenTodos();
                    return Lines(todos.Count == 0 ? "Nothing left to do" : $"{todos.Count} open to-do{Plural(todos.Count)}", todos.Select(t => $"○  {t.Task}"));
                case "complete_todo":
                    return Delete("todo", Arg("todo"), context, stores.CompleteTodo, "Done ✓");
                case "take_note":
                    stores.AddNote(Arg("text"));
                    return new Outcome(true, "Noted");
                case "show_notes":
                    var notes = stores.Notes();
                    return Lines(notes.Count == 0 ? "No notes yet" : $"{notes.Count} note{Plural(notes.Count)}",
                        notes.Select(n => $"{n.Created.ToString("d MMM", Invariant)}  {n.Text}"));
                case "calculate":
                    var result = Calculator.Evaluate(Arg("text"));
                    if (result is null)
                    {
                        Shell($"https://www.google.com/search?q={Uri.EscapeDataString(Arg("text"))}");
                        return new Outcome(true, "Searching the web for that");
                    }
                    Clipboard.SetText(result);
                    return Lines($"{Arg("text")} = {result}", ["Copied to the clipboard"]);
                case "world_time":
                    var clock = WorldClock.At(Arg("text"), DateTime.UtcNow);
                    if (clock is null)
                    {
                        Shell($"https://www.google.com/search?q={Uri.EscapeDataString($"time in {Arg("text")}")}");
                        return new Outcome(true, $"Looking up the time in {Arg("text")}");
                    }
                    var (city, local, zone) = clock.Value;
                    var offset = zone.GetUtcOffset(DateTime.UtcNow);
                    return Lines($"{city}: {local.ToString("h:mm tt", Invariant)}",
                        [$"{local.ToString("dddd d MMMM", Invariant)} · UTC{(offset >= TimeSpan.Zero ? "+" : "-")}{offset:hh\\:mm}"]);
                case "weather":
                    var place = Arg("text").Length > 0 ? Arg("text") : settings.HomeCity.Length > 0 ? settings.HomeCity : Weather.DefaultCity();
                    var (ok, headline, lines) = await Weather.Forecast(place, plan.When?.When).ConfigureAwait(true);
                    return ok ? Lines(headline, lines) : new Outcome(false, headline);
                case "directions":
                    Shell($"https://www.google.com/maps/dir/?api=1&destination={Uri.EscapeDataString(Arg("text"))}");
                    return new Outcome(true, $"Directions to {Arg("text")}");
                case "today":
                    return Lines(now.ToString("dddd d MMMM yyyy", Invariant), [now.ToString("h:mm tt", Invariant)]);
                case "battery":
                    return Battery();
                case "screenshot":
                    return Screenshot();
                case "lock":
                    Native.LockWorkStation();
                    return new Outcome(true, "Locked");
                case "power":
                    return Power(Arg("power"));
                case "compose_email":
                    Shell($"mailto:?subject={Uri.EscapeDataString(Capitalize(Arg("text")))}");
                    return new Outcome(true, "New email draft");
                case "ask_ai":
                    return AskAi(Arg("service"), Arg("text"));
                case "play_music":
                    var query = Arg("text");
                    if (Desktop.Apps.Any(a => a.Name.Equals("Spotify", StringComparison.OrdinalIgnoreCase)))
                    {
                        Shell($"spotify:search:{Uri.EscapeDataString(query)}");
                    }
                    else
                    {
                        Shell($"https://music.youtube.com/search?q={Uri.EscapeDataString(query)}");
                    }
                    return new Outcome(true, $"Finding {query}");
                default:
                    return new Outcome(false, "I need more detail to do that");
            }
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Log.Error($"{plan.Skill.Key} failed", error);
            return new Outcome(false, $"That didn't work: {error.Message}");
        }
    }

    private static string Plural(int count) => count == 1 ? "" : "s";
    private static string Capitalize(string text) => text.Length == 0 ? text : char.ToUpperInvariant(text[0]) + text[1..];
    private static Outcome Lines(string headline, IEnumerable<string> lines) => new(true, headline, lines.ToList());

    private static void Shell(string target) =>
        Process.Start(new ProcessStartInfo(target) { UseShellExecute = true });

    private static Outcome Delete(string slot, string label, CommandContext context, Func<long, bool> delete, string message) =>
        context.StoreIds.TryGetValue($"{slot}:{label}", out var id) && delete(id)
            ? new Outcome(true, message)
            : new Outcome(false, $"Couldn't find that {slot}");

    private Outcome Agenda(DateTime day, DateTime now)
    {
        var events = stores.Events(day.Date, day.Date.AddDays(1));
        var label = (day.Date - now.Date).Days switch { 0 => "Today", 1 => "Tomorrow", _ => day.ToString("dddd d MMM", Invariant) };
        return Lines(events.Count == 0 ? $"{label}: nothing scheduled" : $"{label}: {events.Count} event{Plural(events.Count)}",
            events.Select(e => $"{e.Starts.ToString("h:mm tt", Invariant)}  {Capitalize(e.Title)}"));
    }

    /// <summary>Bring a window forward without synthesizing keys (an Alt tap opens Office key tips).</summary>
    public static void Focus(nint hwnd)
    {
        if (hwnd == 0 || !Native.IsWindow(hwnd))
        {
            return;
        }
        if (Native.IsIconic(hwnd))
        {
            Native.ShowWindow(hwnd, Native.SW_RESTORE);
        }
        var foreground = Native.GetWindowThreadProcessId(Native.GetForegroundWindow(), out _);
        var mine = Native.GetCurrentThreadId();
        Native.AttachThreadInput(mine, foreground, true);
        Native.BringWindowToTop(hwnd);
        Native.SetForegroundWindow(hwnd);
        Native.AttachThreadInput(mine, foreground, false);
    }

    private static Outcome WindowAction(string skill, string label, CommandContext context)
    {
        var window = label.StartsWith("Current window", StringComparison.Ordinal)
            ? context.Windows.FirstOrDefault(w => w.Handle == context.Previous)
            : context.Windows.FirstOrDefault(w => w.Label == label);
        if (window is null || !Native.IsWindow(window.Handle))
        {
            return new Outcome(false, "That window is no longer open");
        }
        switch (skill)
        {
            case "switch_window":
                Focus(window.Handle);
                return new Outcome(true, $"Switched to {window.Title}");
            case "close_window":
                Native.PostMessage(window.Handle, Native.WM_CLOSE, 0, 0);
                return new Outcome(true, $"Closing {window.Title}");
            case "minimize_window":
                Native.ShowWindow(window.Handle, Native.SW_MINIMIZE);
                return new Outcome(true, $"Minimized {window.Title}");
            default:
                Native.ShowWindow(window.Handle, Native.SW_MAXIMIZE);
                Focus(window.Handle);
                return new Outcome(true, $"Maximized {window.Title}");
        }
    }

    private static async Task<Outcome> Shortcut(string name, nint previous)
    {
        var (_, modifiers, key) = Catalogue.Shared.Shortcuts.First(s => s.Name == name);
        var codes = modifiers.Select(m => m switch { "ctrl" => (ushort)0x11, "shift" => (ushort)0x10, "alt" => (ushort)0x12, _ => (ushort)0x5B }).ToList();
        codes.Add(key switch
        {
            "f5" => 0x74,
            "plus" => 0xBB,
            "minus" => 0xBD,
            "tab" => 0x09,
            "period" => 0xBE,
            "right" => 0x27,
            "left" => 0x25,
            _ => char.ToUpperInvariant(key[0]),
        });
        Focus(previous);
        await Task.Delay(120).ConfigureAwait(true);
        Native.SendKeys([.. codes]);
        return new Outcome(true, name);
    }

    private static Outcome Brightness(int change)
    {
        using var current = new ManagementObjectSearcher("root\\WMI", "SELECT CurrentBrightness FROM WmiMonitorBrightness");
        var level = current.Get().Cast<ManagementObject>().Select(o => (int)(byte)o["CurrentBrightness"]).FirstOrDefault(-1);
        if (level < 0)
        {
            return new Outcome(false, "This display's brightness can't be set from Windows");
        }
        var next = Math.Clamp(level + change, 0, 100);
        using var methods = new ManagementObjectSearcher("root\\WMI", "SELECT * FROM WmiMonitorBrightnessMethods");
        foreach (var monitor in methods.Get().Cast<ManagementObject>())
        {
            monitor.InvokeMethod("WmiSetBrightness", [1, (byte)next]);
        }
        return new Outcome(true, $"Brightness {next}%");
    }

    private static async Task<Outcome> Radio(string choice)
    {
        var access = await Windows.Devices.Radios.Radio.RequestAccessAsync();
        if (access != Windows.Devices.Radios.RadioAccessStatus.Allowed)
        {
            return new Outcome(false, "Windows didn't allow changing the radios");
        }
        var kind = choice.StartsWith("Wi-Fi", StringComparison.Ordinal) ? Windows.Devices.Radios.RadioKind.WiFi : Windows.Devices.Radios.RadioKind.Bluetooth;
        var on = choice.EndsWith(" on", StringComparison.Ordinal);
        var radios = await Windows.Devices.Radios.Radio.GetRadiosAsync();
        var radio = radios.FirstOrDefault(r => r.Kind == kind);
        if (radio is null)
        {
            return new Outcome(false, $"No {(kind == Windows.Devices.Radios.RadioKind.WiFi ? "Wi-Fi" : "Bluetooth")} radio found");
        }
        var result = await radio.SetStateAsync(on ? Windows.Devices.Radios.RadioState.On : Windows.Devices.Radios.RadioState.Off);
        return result == Windows.Devices.Radios.RadioAccessStatus.Allowed ? new Outcome(true, choice) : new Outcome(false, "Windows refused the change");
    }

    private static Outcome Theme(bool dark)
    {
        const string key = @"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize";
        Registry.SetValue(key, "AppsUseLightTheme", dark ? 0 : 1, RegistryValueKind.DWord);
        Registry.SetValue(key, "SystemUsesLightTheme", dark ? 0 : 1, RegistryValueKind.DWord);
        // Tell open apps; the system marshals the string for WM_SETTINGCHANGE. Off the UI thread:
        // a broadcast waits on every top-level window.
        _ = Task.Run(() => Native.SendMessageTimeout(0xFFFF /* HWND_BROADCAST */, Native.WM_SETTINGCHANGE, 0,
            "ImmersiveColorSet", 0x0002 /* SMTO_ABORTIFHUNG */, 200, out _));
        return new Outcome(true, dark ? "Dark mode on" : "Light mode on");
    }

    private static Outcome Battery()
    {
        if (!Native.GetSystemPowerStatus(out var status) || status.BatteryFlag == 128)
        {
            return Lines("No battery", ["Running on mains power"]);
        }
        var plugged = status.ACLineStatus == 1;
        var left = status.BatteryLifeTime > 0 ? $" · about {Plan.FormatDuration(status.BatteryLifeTime - status.BatteryLifeTime % 60)} left" : "";
        return Lines($"Battery {status.BatteryLifePercent}%", [plugged ? "Plugged in, charging" : $"On battery{left}"]);
    }

    private static Outcome Screenshot()
    {
        var bounds = System.Windows.Forms.SystemInformation.VirtualScreen;
        using var bitmap = new System.Drawing.Bitmap(bounds.Width, bounds.Height);
        using (var graphics = System.Drawing.Graphics.FromImage(bitmap))
        {
            graphics.CopyFromScreen(bounds.Left, bounds.Top, 0, 0, bitmap.Size);
        }
        var folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "Screenshots");
        Directory.CreateDirectory(folder);
        var file = Path.Combine(folder, $"Jevlet {DateTime.Now.ToString("yyyy-MM-dd HHmmss", Invariant)}.png");
        bitmap.Save(file, System.Drawing.Imaging.ImageFormat.Png);
        return new Outcome(true, $"Saved to Pictures\\Screenshots");
    }

    private static Outcome Power(string action)
    {
        switch (action)
        {
            case "Sleep":
                Native.SetSuspendState(false, false, false);
                return new Outcome(true, "Sleeping");
            case "Restart":
                Process.Start(new ProcessStartInfo("shutdown", "/r /t 0") { CreateNoWindow = true, UseShellExecute = false });
                return new Outcome(true, "Restarting");
            case "Shut down":
                Process.Start(new ProcessStartInfo("shutdown", "/s /t 0") { CreateNoWindow = true, UseShellExecute = false });
                return new Outcome(true, "Shutting down");
            default:
                Process.Start(new ProcessStartInfo("shutdown", "/l") { CreateNoWindow = true, UseShellExecute = false });
                return new Outcome(true, "Signing out");
        }
    }

    private static Outcome AskAi(string service, string request)
    {
        var encoded = Uri.EscapeDataString(request);
        Clipboard.SetText(request);
        Shell(service switch
        {
            "ChatGPT" => $"https://chatgpt.com/?q={encoded}",
            "Gemini" => "https://gemini.google.com/app",
            _ => $"https://claude.ai/new?q={encoded}",
        });
        return new Outcome(true, service == "Gemini" ? "Opening Gemini (request copied, paste it)" : $"Asking {service}");
    }
}
