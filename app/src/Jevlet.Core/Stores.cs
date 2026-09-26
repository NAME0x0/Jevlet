using System.Globalization;
using Microsoft.Data.Sqlite;

namespace Jevlet.Core;

public sealed record CalendarEvent(long Id, string Title, DateTime Starts);
public sealed record Alarm(long Id, TimeSpan Time, string Label, bool Enabled);
public sealed record Reminder(long Id, string Task, DateTime Due, bool Done);
public sealed record Todo(long Id, string Task, bool Done, DateTime Created);
public sealed record Note(long Id, string Text, DateTime Created);

/// <summary>
/// The user's calendar, alarms, reminders, to-dos, notes, and command history in one local
/// SQLite file. Labels offered to the model use the formats it was trained on.
/// </summary>
public sealed class Stores : IDisposable
{
    private readonly SqliteConnection _connection;

    public Stores(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        _connection = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path }.ToString());
        _connection.Open();
        Execute("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, title TEXT NOT NULL, starts TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS alarms (id INTEGER PRIMARY KEY, time TEXT NOT NULL, label TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, task TEXT NOT NULL, due TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, task TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL, completed TEXT);
            CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, at TEXT NOT NULL, command TEXT NOT NULL,
                skill TEXT NOT NULL, arguments TEXT NOT NULL, gate TEXT NOT NULL, outcome TEXT NOT NULL, rank INTEGER NOT NULL);
            """);
    }

    public static string DefaultPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Jevlet", "jevlet.db");

    private static string Iso(DateTime value) => value.ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
    private static DateTime Parse(string value) => DateTime.ParseExact(value, "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);

    private int Execute(string sql, params (string Name, object? Value)[] parameters)
    {
        using var command = _connection.CreateCommand();
        command.CommandText = sql;
        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value ?? DBNull.Value);
        }
        return command.ExecuteNonQuery();
    }

    private long Insert(string sql, params (string Name, object? Value)[] parameters)
    {
        Execute(sql, parameters);
        using var command = _connection.CreateCommand();
        command.CommandText = "SELECT last_insert_rowid()";
        return (long)command.ExecuteScalar()!;
    }

    private List<T> Query<T>(string sql, Func<SqliteDataReader, T> read, params (string Name, object? Value)[] parameters)
    {
        using var command = _connection.CreateCommand();
        command.CommandText = sql;
        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value ?? DBNull.Value);
        }
        using var reader = command.ExecuteReader();
        var rows = new List<T>();
        while (reader.Read())
        {
            rows.Add(read(reader));
        }
        return rows;
    }

    // ------------------------------------------------------------------ calendar

    public CalendarEvent AddEvent(string title, DateTime starts)
    {
        var id = Insert("INSERT INTO events (title, starts, created) VALUES ($t, $s, $c)",
            ("$t", title), ("$s", Iso(starts)), ("$c", Iso(DateTime.Now)));
        return new CalendarEvent(id, title, starts);
    }

    public List<CalendarEvent> Events(DateTime from, DateTime until) =>
        Query("SELECT id, title, starts FROM events WHERE starts >= $f AND starts < $u ORDER BY starts",
            r => new CalendarEvent(r.GetInt64(0), r.GetString(1), Parse(r.GetString(2))),
            ("$f", Iso(from)), ("$u", Iso(until)));

    public List<CalendarEvent> UpcomingEvents(DateTime now, int limit = 20) =>
        Query("SELECT id, title, starts FROM events WHERE starts >= $n ORDER BY starts LIMIT $l",
            r => new CalendarEvent(r.GetInt64(0), r.GetString(1), Parse(r.GetString(2))),
            ("$n", Iso(now.Date)), ("$l", limit));

    public bool MoveEvent(long id, DateTime starts) =>
        Execute("UPDATE events SET starts = $s WHERE id = $i", ("$s", Iso(starts)), ("$i", id)) == 1;

    public bool CancelEvent(long id) => Execute("DELETE FROM events WHERE id = $i", ("$i", id)) == 1;

    // ------------------------------------------------------------------ alarms

    public Alarm AddAlarm(TimeSpan time, string label)
    {
        var id = Insert("INSERT INTO alarms (time, label) VALUES ($t, $l)",
            ("$t", time.ToString(@"hh\:mm", CultureInfo.InvariantCulture)), ("$l", label));
        return new Alarm(id, time, label, true);
    }

    public List<Alarm> Alarms() =>
        Query("SELECT id, time, label, enabled FROM alarms ORDER BY time",
            r => new Alarm(r.GetInt64(0), TimeSpan.ParseExact(r.GetString(1), @"hh\:mm", CultureInfo.InvariantCulture),
                r.GetString(2), r.GetInt64(3) == 1));

    public bool CancelAlarm(long id) => Execute("DELETE FROM alarms WHERE id = $i", ("$i", id)) == 1;

    // ------------------------------------------------------------------ reminders

    public Reminder AddReminder(string task, DateTime due)
    {
        var id = Insert("INSERT INTO reminders (task, due) VALUES ($t, $d)", ("$t", task), ("$d", Iso(due)));
        return new Reminder(id, task, due, false);
    }

    public List<Reminder> PendingReminders() =>
        Query("SELECT id, task, due, done FROM reminders WHERE done = 0 ORDER BY due",
            r => new Reminder(r.GetInt64(0), r.GetString(1), Parse(r.GetString(2)), false));

    public List<Reminder> DueReminders(DateTime now) =>
        PendingReminders().Where(r => r.Due <= now).ToList();

    public bool FinishReminder(long id) => Execute("UPDATE reminders SET done = 1 WHERE id = $i", ("$i", id)) == 1;

    public bool CancelReminder(long id) => Execute("DELETE FROM reminders WHERE id = $i", ("$i", id)) == 1;

    // ------------------------------------------------------------------ to-dos and notes

    public Todo AddTodo(string task)
    {
        var id = Insert("INSERT INTO todos (task, created) VALUES ($t, $c)", ("$t", task), ("$c", Iso(DateTime.Now)));
        return new Todo(id, task, false, DateTime.Now);
    }

    public List<Todo> OpenTodos() =>
        Query("SELECT id, task, done, created FROM todos WHERE done = 0 ORDER BY created",
            r => new Todo(r.GetInt64(0), r.GetString(1), false, Parse(r.GetString(3))));

    public bool CompleteTodo(long id) =>
        Execute("UPDATE todos SET done = 1, completed = $c WHERE id = $i", ("$c", Iso(DateTime.Now)), ("$i", id)) == 1;

    public Note AddNote(string text)
    {
        var id = Insert("INSERT INTO notes (text, created) VALUES ($t, $c)", ("$t", text), ("$c", Iso(DateTime.Now)));
        return new Note(id, text, DateTime.Now);
    }

    public List<Note> Notes(int limit = 50) =>
        Query("SELECT id, text, created FROM notes ORDER BY created DESC LIMIT $l",
            r => new Note(r.GetInt64(0), r.GetString(1), Parse(r.GetString(2))), ("$l", limit));

    // ------------------------------------------------------------------ history

    /// <summary>What ran and whether the user accepted it (rank 0) or chose an alternative.</summary>
    public void Record(string command, string skill, string arguments, string gate, string outcome, int rank) =>
        Execute("INSERT INTO history (at, command, skill, arguments, gate, outcome, rank) VALUES ($a, $c, $s, $g2, $g, $o, $r)",
            ("$a", Iso(DateTime.Now)), ("$c", command), ("$s", skill), ("$g2", arguments), ("$g", gate), ("$o", outcome), ("$r", rank));

    // ------------------------------------------------------------------ model labels

    private static string Day(DateTime when, DateTime now) => (when.Date - now.Date).Days switch
    {
        0 => "today",
        1 => "tomorrow",
        _ => when.ToString("ddd", CultureInfo.InvariantCulture),
    };

    private static string Capitalize(string text) =>
        text.Length == 0 ? text : char.ToUpperInvariant(text[0]) + text[1..];

    public static string EventLabel(CalendarEvent item, DateTime now) =>
        $"{Capitalize(item.Title)} · {Day(item.Starts, now)} {item.Starts:HH:mm}";

    public static string AlarmLabel(Alarm alarm) =>
        $"{alarm.Time:hh\\:mm} · {Capitalize(alarm.Label.Length > 0 ? alarm.Label : "Alarm")}";

    public static string ReminderLabel(Reminder reminder, DateTime now) =>
        $"{Capitalize(reminder.Task)} · {Day(reminder.Due, now)} {reminder.Due:HH:mm}";

    /// <summary>
    /// The store pools of a snapshot, and how to map a chosen label back to its row: ids are
    /// keyed "slot:label" (slot is event, alarm, reminder, or todo).
    /// </summary>
    public (Snapshot Snapshot, IReadOnlyDictionary<string, long> Ids) Fill(Snapshot snapshot, DateTime now)
    {
        var ids = new Dictionary<string, long>(StringComparer.Ordinal);
        List<string> Labels<T>(string slot, IEnumerable<T> rows, Func<T, string> label, Func<T, long> id)
        {
            var labels = new List<string>();
            foreach (var row in rows)
            {
                var text = label(row);
                if (ids.TryAdd($"{slot}:{text}", id(row)))
                {
                    labels.Add(text);
                }
            }
            return labels;
        }
        var filled = snapshot with
        {
            Events = Labels("event", UpcomingEvents(now), e => EventLabel(e, now), e => e.Id),
            Alarms = Labels("alarm", Alarms(), AlarmLabel, a => a.Id),
            Reminders = Labels("reminder", PendingReminders(), r => ReminderLabel(r, now), r => r.Id),
            Todos = Labels("todo", OpenTodos(), t => t.Task, t => t.Id),
        };
        return (filled, ids);
    }

    public void Dispose() => _connection.Dispose();
}
