using System.Globalization;
using System.Text.RegularExpressions;

namespace Jevlet.Core;

/// <summary>A resolved date/time and how it was said.</summary>
public sealed record TimeMatch(DateTime When, bool HasDate, bool HasClock)
{
    /// <summary>"today 6:00 PM", "tomorrow 9:00 AM", "Tue 30 Sep 4:00 PM".</summary>
    public string Describe(DateTime now)
    {
        var clock = When.ToString("h:mm tt", CultureInfo.InvariantCulture);
        var days = (When.Date - now.Date).Days;
        var day = days switch
        {
            0 => "today",
            1 => "tomorrow",
            > 1 and < 7 => When.ToString("dddd", CultureInfo.InvariantCulture),
            _ => When.ToString("ddd d MMM", CultureInfo.InvariantCulture),
        };
        return $"{day} {clock}";
    }
}

/// <summary>
/// Dates and times in assistant phrasing, resolved against "now". Parsed by rules, never
/// predicted: the model picks the skill and the span; this turns "tomorrow at 6" into a time.
/// </summary>
public static class TimeParser
{
    private const RegexOptions Options = RegexOptions.IgnoreCase | RegexOptions.CultureInvariant;

    private static readonly string[] WeekdayNames = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
    private static readonly string[] MonthNames = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
    private static readonly Dictionary<string, int> NumberWords = new(StringComparer.Ordinal)
    {
        ["a"] = 1, ["an"] = 1, ["one"] = 1, ["two"] = 2, ["three"] = 3, ["four"] = 4, ["five"] = 5, ["six"] = 6,
        ["seven"] = 7, ["eight"] = 8, ["nine"] = 9, ["ten"] = 10, ["eleven"] = 11, ["twelve"] = 12,
        ["fifteen"] = 15, ["twenty"] = 20, ["thirty"] = 30, ["forty"] = 40, ["forty five"] = 45, ["fifty"] = 50,
    };

    private static readonly Regex TextSpeak = new(@"\b(tmrw|tmr|tmrrw|2moro|2morrow)\b", Options);
    private static readonly Regex Relative = new(
        @"\bin\s+(?:(half an hour)|(an hour and a half)|(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty five|forty|fifty)\s*(seconds?|secs?|minutes?|mins?|m|hours?|hrs?|h|days?|weeks?))\b",
        Options);
    private static readonly Regex DayAfterTomorrow = new(@"\bday after tomorrow\b", Options);
    private static readonly Regex Tomorrow = new(@"\btomorrow\b", Options);
    private static readonly Regex Today = new(@"\b(today|tonight|this (?:morning|afternoon|evening))\b", Options);
    // Full names always count; abbreviations only after a lead-in ("on sat", not "I sat down").
    private static readonly Regex Weekday = new(
        @"\b(?:(next|this|on|coming)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b(next|this|on|coming)\s+(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        Options);
    private static readonly Regex NextWeek = new(@"\bnext week\b", Options);
    private static readonly Regex Weekend = new(@"\b(?:this |next )?weekend\b", Options);
    private static readonly Regex Ordinal = new(@"\b(?:on )?the (\d{1,2})(?:st|nd|rd|th)?\b|\b(\d{1,2})(?:st|nd|rd|th)\b", Options);
    private static readonly Regex MonthDay = new(
        @"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        Options);
    private static readonly Regex Noon = new(@"\b(noon|midday|midnight)\b", Options);
    private static readonly Regex HalfPast = new(@"\b(half past|quarter past|quarter to)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", Options);
    private static readonly Regex Clock = new(
        @"(?<![\w:.])(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?|o'clock)?(?![\d%]|\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|st|nd|rd|th|percent|km|kg|miles?|times|x)\b)",
        Options);
    private static readonly Regex ClockContext = new(@"(\b(at|by|around|for|until|till|from|to)|@)\s*$", Options);
    // Context for an ambiguous "at 7" (explicit am/pm is read from the time itself; "I am"
    // must not count as a morning hint).
    private static readonly Regex Evening = new(@"\b(tonight|evening|night|afternoon|dinner|supper)\b", Options);
    private static readonly Regex Morning = new(@"\b(morning|wake|waking|breakfast|sunrise)\b", Options);

    public static bool Mentions(string command) => Parse(command, DateTime.Now) is not null;

    public static TimeMatch? Parse(string command, DateTime now)
    {
        var text = TextSpeak.Replace(command.ToLowerInvariant(), "tomorrow");
        var relative = Relative.Match(text);
        if (relative.Success)
        {
            var span = RelativeSpan(relative);
            if (span is not null)
            {
                return new TimeMatch(now + span.Value, true, true);
            }
        }

        var (date, hint) = FindDate(text, now);
        var clock = FindClock(text, hint);
        if (date is null && clock is null)
        {
            return null;
        }
        if (clock is null)
        {
            var defaultHour = hint switch
            {
                "evening" when text.Contains("tonight", StringComparison.Ordinal) => 20,
                "evening" => 19,
                "afternoon" => 15,
                _ => 9,
            };
            var when = date!.Value.Date.AddHours(defaultHour);
            if (when <= now && date.Value.Date == now.Date)
            {
                when = now.AddMinutes(1);
            }
            return new TimeMatch(when, true, false);
        }
        var (hour, minute, meridiem) = clock.Value;
        if (date is not null)
        {
            var day = date.Value.Date;
            return new TimeMatch(day.Add(ResolveHour(hour, minute, meridiem, hint, day, now)), true, true);
        }
        // Time only: the next occurrence, today or tomorrow.
        var today = now.Date.Add(ResolveHour(hour, minute, meridiem, hint, now.Date, now));
        return new TimeMatch(today > now ? today : now.Date.AddDays(1).Add(ResolveHour(hour, minute, meridiem, hint, now.Date.AddDays(1), now)), false, true);
    }

    private static TimeSpan? RelativeSpan(Match match)
    {
        if (match.Groups[1].Success)
        {
            return TimeSpan.FromMinutes(30);
        }
        if (match.Groups[2].Success)
        {
            return TimeSpan.FromMinutes(90);
        }
        var word = match.Groups[3].Value;
        double amount = double.TryParse(word, NumberStyles.Float, CultureInfo.InvariantCulture, out var number)
            ? number
            : NumberWords.TryGetValue(word, out var named) ? named : 0;
        var unit = match.Groups[4].Value;
        return unit[0] switch
        {
            's' => TimeSpan.FromSeconds(amount),
            'm' => TimeSpan.FromMinutes(amount),
            'h' => TimeSpan.FromHours(amount),
            'd' => TimeSpan.FromDays(amount),
            'w' => TimeSpan.FromDays(7 * amount),
            _ => null,
        };
    }

    private static (DateTime? Date, string Hint) FindDate(string text, DateTime now)
    {
        var hint = Evening.IsMatch(text) ? "evening" : Morning.IsMatch(text) ? "morning" : "";
        if (text.Contains("afternoon", StringComparison.Ordinal))
        {
            hint = "afternoon";
        }
        if (DayAfterTomorrow.IsMatch(text))
        {
            return (now.Date.AddDays(2), hint);
        }
        if (Tomorrow.IsMatch(text))
        {
            return (now.Date.AddDays(1), hint);
        }
        if (Today.IsMatch(text))
        {
            return (now.Date, hint);
        }
        var monthDay = MonthDay.Match(text);
        if (monthDay.Success)
        {
            var dayText = monthDay.Groups[1].Success ? monthDay.Groups[1].Value : monthDay.Groups[4].Value;
            var monthText = monthDay.Groups[2].Success ? monthDay.Groups[2].Value : monthDay.Groups[3].Value;
            var month = Array.FindIndex(MonthNames, m => m.StartsWith(monthText[..3], StringComparison.Ordinal)) + 1;
            var day = int.Parse(dayText, CultureInfo.InvariantCulture);
            if (month > 0 && day is >= 1 and <= 31)
            {
                var year = now.Year;
                var date = SafeDate(year, month, day);
                if (date < now.Date)
                {
                    date = SafeDate(year + 1, month, day);
                }
                return (date, hint);
            }
        }
        var weekday = Weekday.Match(text);
        if (weekday.Success)
        {
            var name = weekday.Groups[2].Success ? weekday.Groups[2].Value : weekday.Groups[4].Value;
            var lead = weekday.Groups[1].Success ? weekday.Groups[1].Value : weekday.Groups[3].Value;
            var target = Array.FindIndex(WeekdayNames, n => n.StartsWith(name[..3], StringComparison.Ordinal));
            int ahead;
            if (NextWeek.IsMatch(text))
            {
                // "next week on thursday": that weekday in the week starting next Monday.
                static int MondayBased(int day) => day == 0 ? 7 : day;
                ahead = 8 - MondayBased((int)now.DayOfWeek) + (MondayBased(target) - 1);
            }
            else
            {
                ahead = (target - (int)now.DayOfWeek + 7) % 7;
                if (ahead == 0 && lead == "next")
                {
                    ahead = 7;
                }
            }
            return (now.Date.AddDays(ahead), hint);
        }
        if (Weekend.IsMatch(text))
        {
            var ahead = ((int)DayOfWeek.Saturday - (int)now.DayOfWeek + 7) % 7;
            return (now.Date.AddDays(ahead), hint);
        }
        if (NextWeek.IsMatch(text))
        {
            return (now.Date.AddDays(7), hint);
        }
        var ordinal = Ordinal.Match(text);
        if (ordinal.Success)
        {
            var day = int.Parse(ordinal.Groups[1].Success ? ordinal.Groups[1].Value : ordinal.Groups[2].Value, CultureInfo.InvariantCulture);
            if (day is >= 1 and <= 31)
            {
                var date = SafeDate(now.Year, now.Month, day);
                if (date < now.Date)
                {
                    var next = now.AddMonths(1);
                    date = SafeDate(next.Year, next.Month, day);
                }
                return (date, hint);
            }
        }
        return (null, hint);
    }

    private static DateTime SafeDate(int year, int month, int day) =>
        new(year, month, Math.Min(day, DateTime.DaysInMonth(year, month)), 0, 0, 0, DateTimeKind.Local);

    private static (int Hour, int Minute, string Meridiem)? FindClock(string text, string hint)
    {
        var noon = Noon.Match(text);
        if (noon.Success)
        {
            return noon.Value == "midnight" ? (0, 0, "am") : (12, 0, "pm");
        }
        var half = HalfPast.Match(text);
        if (half.Success)
        {
            var word = half.Groups[2].Value;
            var hour = int.TryParse(word, NumberStyles.Integer, CultureInfo.InvariantCulture, out var h) ? h : NumberWords[word];
            return half.Groups[1].Value switch
            {
                "half past" => (hour, 30, ""),
                "quarter past" => (hour, 15, ""),
                _ => (hour == 1 ? 12 : hour == 13 ? 12 : hour - 1, 45, ""),
            };
        }
        foreach (Match match in Clock.Matches(text))
        {
            var hour = int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
            var hasMinutes = match.Groups[2].Success;
            var minute = hasMinutes ? int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture) : 0;
            var suffix = match.Groups[3].Value.Replace(".", "", StringComparison.Ordinal);
            var before = text[..match.Index];
            var after = text[(match.Index + match.Length)..].TrimStart();
            var inContext = ClockContext.IsMatch(before)
                || suffix.Length > 0
                || hasMinutes
                || after.StartsWith("in the ", StringComparison.Ordinal)
                || after.StartsWith("tonight", StringComparison.Ordinal)
                || after.StartsWith("tomorrow", StringComparison.Ordinal);
            if (!inContext || hour > 23 || minute > 59)
            {
                continue;
            }
            var meridiem = suffix is "am" or "pm" ? suffix : "";
            if (after.StartsWith("in the morning", StringComparison.Ordinal))
            {
                meridiem = "am";
            }
            else if (after.StartsWith("in the afternoon", StringComparison.Ordinal) || after.StartsWith("in the evening", StringComparison.Ordinal) || after.StartsWith("tonight", StringComparison.Ordinal))
            {
                meridiem = "pm";
            }
            return (hour, minute, meridiem);
        }
        return null;
    }

    private static TimeSpan ResolveHour(int hour, int minute, string meridiem, string hint, DateTime day, DateTime now)
    {
        if (meridiem == "am")
        {
            return new TimeSpan(hour % 12, minute, 0);
        }
        if (meridiem == "pm")
        {
            return new TimeSpan(hour % 12 + 12, minute, 0);
        }
        if (hour == 0 || hour > 12)
        {
            return new TimeSpan(hour % 24, minute, 0); // 24-hour clock
        }
        if (hint is "evening" or "afternoon")
        {
            return new TimeSpan(hour % 12 + 12, minute, 0);
        }
        if (hint == "morning")
        {
            return new TimeSpan(hour % 12, minute, 0);
        }
        // Ambiguous "at 7": on a later day, 7-11 means morning and 12-6 afternoon; today, the
        // next occurrence after now.
        var morning = new TimeSpan(hour % 12, minute, 0);
        var afternoon = new TimeSpan(hour % 12 + 12, minute, 0);
        if (day.Date != now.Date)
        {
            return hour is >= 7 and <= 11 ? morning : afternoon;
        }
        return day.Date.Add(morning) > now ? morning : afternoon;
    }
}
