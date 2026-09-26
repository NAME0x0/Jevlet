using System.Windows.Threading;
using Jevlet.Core;

namespace Jevlet.App;

internal sealed class RunningTimer(string label, TimeSpan length)
{
    public string Label { get; } = label;
    public TimeSpan Length { get; } = length;
    public DateTime? Due { get; set; } = DateTime.Now + length;
    public TimeSpan? PausedRemaining { get; set; }
    public TimeSpan Remaining => PausedRemaining ?? (Due - DateTime.Now) ?? TimeSpan.Zero;
}

/// <summary>
/// Everything time-driven: reminders coming due, alarms ringing (with snooze and stop), countdown
/// timers, and the stopwatch. Ticks every second on the UI thread; state that must survive a
/// restart (alarms, reminders) lives in the stores.
/// </summary>
internal sealed class Scheduler
{
    private readonly Stores _stores;
    private readonly Notifier _notifier;
    private readonly DispatcherTimer _tick = new() { Interval = TimeSpan.FromSeconds(1) };
    private readonly HashSet<(long Alarm, DateTime Minute)> _rung = [];
    private DateTime? _snoozeUntil;
    private string _snoozeLabel = "";
    private DateTime? _stopwatchStarted;
    private TimeSpan _stopwatchBanked;

    public List<RunningTimer> Timers { get; } = [];
    public string? RingingLabel { get; private set; }

    public Scheduler(Stores stores, Notifier notifier)
    {
        _stores = stores;
        _notifier = notifier;
        _tick.Tick += (_, _) => Tick(DateTime.Now);
        _tick.Start();
    }

    public TimeSpan Stopwatch => _stopwatchBanked + (_stopwatchStarted is { } started ? DateTime.Now - started : TimeSpan.Zero);
    public bool StopwatchRunning => _stopwatchStarted is not null;

    private void Tick(DateTime now)
    {
        foreach (var reminder in _stores.DueReminders(now))
        {
            _stores.FinishReminder(reminder.Id);
            _notifier.Show("Reminder", Capitalize(reminder.Task), Notifier.Kind.Reminder);
        }
        var minute = new DateTime(now.Year, now.Month, now.Day, now.Hour, now.Minute, 0, DateTimeKind.Local);
        foreach (var alarm in _stores.Alarms().Where(a => a.Enabled))
        {
            if (alarm.Time.Hours == now.Hour && alarm.Time.Minutes == now.Minute && _rung.Add((alarm.Id, minute)))
            {
                Ring(alarm.Label.Length > 0 ? Capitalize(alarm.Label) : "Alarm");
            }
        }
        if (_snoozeUntil is { } snooze && now >= snooze)
        {
            _snoozeUntil = null;
            Ring(_snoozeLabel);
        }
        foreach (var timer in Timers.ToList())
        {
            if (timer.PausedRemaining is null && timer.Due <= now)
            {
                Timers.Remove(timer);
                Ring($"{timer.Label} is done");
            }
        }
    }

    private void Ring(string label)
    {
        RingingLabel = label;
        Log.Info($"ringing: {label}");
        _notifier.Ring(label, Snooze, StopRinging);
    }

    public string Snooze()
    {
        if (RingingLabel is null)
        {
            return "Nothing is ringing";
        }
        _snoozeLabel = RingingLabel;
        _snoozeUntil = DateTime.Now.AddMinutes(9);
        StopRinging();
        return "Snoozed for 9 minutes";
    }

    public string StopRinging()
    {
        var had = RingingLabel is not null;
        RingingLabel = null;
        _notifier.StopRinging();
        return had ? "Alarm stopped" : "Nothing is ringing";
    }

    public RunningTimer StartTimer(TimeSpan length)
    {
        var timer = new RunningTimer($"{Plan.FormatDuration((int)length.TotalSeconds)} timer", length);
        Timers.Add(timer);
        return timer;
    }

    /// <summary>Pause, resume, cancel, time left, or add time to the newest timer.</summary>
    public string ControlTimer(string action, int? addSeconds)
    {
        var timer = Timers.LastOrDefault();
        if (timer is null)
        {
            return "No timer is running";
        }
        switch (action)
        {
            case "Pause" when timer.PausedRemaining is null:
                timer.PausedRemaining = timer.Remaining;
                timer.Due = null;
                return $"Paused with {Plan.FormatDuration((int)timer.Remaining.TotalSeconds)} left";
            case "Resume" when timer.PausedRemaining is { } left:
                timer.Due = DateTime.Now + left;
                timer.PausedRemaining = null;
                return "Timer resumed";
            case "Cancel":
                Timers.Remove(timer);
                return "Timer cancelled";
            case "Add time":
                var extra = TimeSpan.FromSeconds(addSeconds ?? 60);
                if (timer.PausedRemaining is { } paused)
                {
                    timer.PausedRemaining = paused + extra;
                }
                else
                {
                    timer.Due += extra;
                }
                return $"Added {Plan.FormatDuration((int)extra.TotalSeconds)}";
            default:
                return $"{Plan.FormatDuration(Math.Max(1, (int)timer.Remaining.TotalSeconds))} left";
        }
    }

    public string ControlStopwatch(string action)
    {
        switch (action)
        {
            case "Start":
                _stopwatchStarted ??= DateTime.Now;
                return "Stopwatch running";
            case "Stop":
                if (_stopwatchStarted is { } started)
                {
                    _stopwatchBanked += DateTime.Now - started;
                    _stopwatchStarted = null;
                }
                return $"Stopwatch: {Stopwatch:hh\\:mm\\:ss\\.f}";
            default:
                _stopwatchStarted = null;
                _stopwatchBanked = TimeSpan.Zero;
                return "Stopwatch reset";
        }
    }

    private static string Capitalize(string text) => text.Length == 0 ? text : char.ToUpperInvariant(text[0]) + text[1..];
}
