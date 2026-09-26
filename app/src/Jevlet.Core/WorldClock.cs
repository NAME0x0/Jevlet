namespace Jevlet.Core;

/// <summary>Local time in a named city (IANA zones; .NET resolves them on Windows through ICU).</summary>
public static class WorldClock
{
    private static readonly Dictionary<string, string> Zones = new(StringComparer.OrdinalIgnoreCase)
    {
        ["london"] = "Europe/London", ["uk"] = "Europe/London", ["england"] = "Europe/London", ["edinburgh"] = "Europe/London",
        ["manchester"] = "Europe/London", ["dublin"] = "Europe/Dublin", ["paris"] = "Europe/Paris", ["france"] = "Europe/Paris",
        ["berlin"] = "Europe/Berlin", ["germany"] = "Europe/Berlin", ["munich"] = "Europe/Berlin", ["madrid"] = "Europe/Madrid",
        ["barcelona"] = "Europe/Madrid", ["rome"] = "Europe/Rome", ["milan"] = "Europe/Rome", ["amsterdam"] = "Europe/Amsterdam",
        ["brussels"] = "Europe/Brussels", ["zurich"] = "Europe/Zurich", ["vienna"] = "Europe/Vienna", ["prague"] = "Europe/Prague",
        ["warsaw"] = "Europe/Warsaw", ["stockholm"] = "Europe/Stockholm", ["oslo"] = "Europe/Oslo", ["copenhagen"] = "Europe/Copenhagen",
        ["helsinki"] = "Europe/Helsinki", ["athens"] = "Europe/Athens", ["istanbul"] = "Europe/Istanbul", ["moscow"] = "Europe/Moscow",
        ["lisbon"] = "Europe/Lisbon", ["dubai"] = "Asia/Dubai", ["abu dhabi"] = "Asia/Dubai", ["sharjah"] = "Asia/Dubai", ["uae"] = "Asia/Dubai",
        ["doha"] = "Asia/Qatar", ["riyadh"] = "Asia/Riyadh", ["jeddah"] = "Asia/Riyadh", ["kuwait city"] = "Asia/Kuwait", ["kuwait"] = "Asia/Kuwait",
        ["muscat"] = "Asia/Muscat", ["bahrain"] = "Asia/Bahrain", ["amman"] = "Asia/Amman", ["beirut"] = "Asia/Beirut", ["cairo"] = "Africa/Cairo",
        ["tehran"] = "Asia/Tehran", ["baghdad"] = "Asia/Baghdad", ["karachi"] = "Asia/Karachi", ["lahore"] = "Asia/Karachi", ["islamabad"] = "Asia/Karachi",
        ["pakistan"] = "Asia/Karachi", ["delhi"] = "Asia/Kolkata", ["new delhi"] = "Asia/Kolkata", ["mumbai"] = "Asia/Kolkata", ["bangalore"] = "Asia/Kolkata",
        ["india"] = "Asia/Kolkata", ["dhaka"] = "Asia/Dhaka", ["colombo"] = "Asia/Colombo", ["kathmandu"] = "Asia/Kathmandu",
        ["bangkok"] = "Asia/Bangkok", ["hanoi"] = "Asia/Bangkok", ["jakarta"] = "Asia/Jakarta", ["singapore"] = "Asia/Singapore",
        ["kuala lumpur"] = "Asia/Kuala_Lumpur", ["manila"] = "Asia/Manila", ["hong kong"] = "Asia/Hong_Kong", ["beijing"] = "Asia/Shanghai",
        ["shanghai"] = "Asia/Shanghai", ["china"] = "Asia/Shanghai", ["taipei"] = "Asia/Taipei", ["seoul"] = "Asia/Seoul", ["tokyo"] = "Asia/Tokyo",
        ["osaka"] = "Asia/Tokyo", ["kyoto"] = "Asia/Tokyo", ["japan"] = "Asia/Tokyo", ["sydney"] = "Australia/Sydney", ["melbourne"] = "Australia/Melbourne",
        ["perth"] = "Australia/Perth", ["auckland"] = "Pacific/Auckland", ["nairobi"] = "Africa/Nairobi", ["lagos"] = "Africa/Lagos",
        ["johannesburg"] = "Africa/Johannesburg", ["cape town"] = "Africa/Johannesburg", ["casablanca"] = "Africa/Casablanca",
        ["new york"] = "America/New_York", ["nyc"] = "America/New_York", ["boston"] = "America/New_York", ["miami"] = "America/New_York",
        ["washington"] = "America/New_York", ["toronto"] = "America/Toronto", ["montreal"] = "America/Toronto", ["chicago"] = "America/Chicago",
        ["houston"] = "America/Chicago", ["dallas"] = "America/Chicago", ["denver"] = "America/Denver", ["los angeles"] = "America/Los_Angeles",
        ["la"] = "America/Los_Angeles", ["san francisco"] = "America/Los_Angeles", ["seattle"] = "America/Los_Angeles",
        ["vancouver"] = "America/Vancouver", ["mexico city"] = "America/Mexico_City", ["sao paulo"] = "America/Sao_Paulo",
        ["buenos aires"] = "America/Argentina/Buenos_Aires", ["lima"] = "America/Lima", ["bogota"] = "America/Bogota",
        ["honolulu"] = "Pacific/Honolulu", ["hawaii"] = "Pacific/Honolulu",
    };

    /// <summary>(city, local time, zone) or null when the city is unknown.</summary>
    public static (string City, DateTime Local, TimeZoneInfo Zone)? At(string place, DateTime utcNow)
    {
        var key = place.Trim().Trim('?', '.', '!').Replace("the ", "", StringComparison.OrdinalIgnoreCase);
        if (!Zones.TryGetValue(key, out var id))
        {
            return null;
        }
        var zone = TimeZoneInfo.FindSystemTimeZoneById(id);
        return (Title(key), TimeZoneInfo.ConvertTimeFromUtc(utcNow, zone), zone);
    }

    private static string Title(string text) =>
        string.Join(' ', text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => w.Length <= 3 && w is "uk" or "uae" or "nyc" or "la" ? w.ToUpperInvariant() : char.ToUpperInvariant(w[0]) + w[1..]));
}
