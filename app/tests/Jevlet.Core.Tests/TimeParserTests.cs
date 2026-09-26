using Jevlet.Core;

namespace Jevlet.Core.Tests;

public class TimeParserTests
{
    // Friday 25 September 2026, 2 pm.
    private static readonly DateTime Now = new(2026, 9, 25, 14, 0, 0, DateTimeKind.Local);

    [Theory]
    [InlineData("set an alarm for 7am", "2026-09-26 07:00")]
    [InlineData("wake me up at 6:15 tomorrow", "2026-09-26 06:15")]
    [InlineData("remind me to water the plants at 6pm", "2026-09-25 18:00")]
    [InlineData("in two hours remind me to call mum", "2026-09-25 16:00")]
    [InlineData("remind me in 45 minutes", "2026-09-25 14:45")]
    [InlineData("in half an hour", "2026-09-25 14:30")]
    [InlineData("put a haircut on friday at 4pm on my calendar", "2026-09-25 16:00")]
    [InlineData("schedule lunch with omar next tuesday at 1", "2026-09-29 13:00")]
    [InlineData("push the dentist appointment to next week", "2026-10-02 09:00")]
    [InlineData("tonight at 8", "2026-09-25 20:00")]
    [InlineData("at noon", "2026-09-26 12:00")]
    [InlineData("on the 15th at 11", "2026-10-15 11:00")]
    [InlineData("half past six", "2026-09-25 18:30")]
    [InlineData("quarter to eight", "2026-09-25 19:45")]
    [InlineData("next week on thursday at 3", "2026-10-01 15:00")]
    [InlineData("this weekend", "2026-09-26 09:00")]
    [InlineData("tomorrow afternoon", "2026-09-26 15:00")]
    [InlineData("I am free at 5", "2026-09-25 17:00")]
    [InlineData("dec 3rd at 9am", "2026-12-03 09:00")]
    [InlineData("3 october", "2026-10-03 09:00")]
    [InlineData("tmrw at 7", "2026-09-26 07:00")]
    [InlineData("alarm at 11:30 tonight", "2026-09-25 23:30")]
    [InlineData("at 18:30", "2026-09-25 18:30")]
    [InlineData("move standup to 10", "2026-09-25 22:00")]
    public void Resolves_assistant_phrasing(string command, string expected)
    {
        var match = TimeParser.Parse(command, Now);
        Assert.NotNull(match);
        Assert.Equal(expected, match.When.ToString("yyyy-MM-dd HH:mm", System.Globalization.CultureInfo.InvariantCulture));
    }

    [Theory]
    [InlineData("what's 18 percent of 240")]
    [InlineData("set a timer for 25 minutes")]
    [InlineData("open spotify")]
    [InlineData("order number 123456")]
    [InlineData("convert 5 miles to km")]
    [InlineData("I sat down with the v2 slides")]
    public void Ignores_numbers_that_are_not_times(string command) =>
        Assert.Null(TimeParser.Parse(command, Now));

    [Fact]
    public void Describes_relative_days()
    {
        Assert.Equal("tomorrow 7:00 AM", TimeParser.Parse("tmrw at 7am", Now)!.Describe(Now));
        Assert.Equal("today 6:00 PM", TimeParser.Parse("at 6pm", Now)!.Describe(Now));
    }
}
