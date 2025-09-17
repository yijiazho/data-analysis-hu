import java.awt.font.TextAttribute;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;


public class WordCount extends Configured implements Tool {

  /**
   * Mapper class:
   * Input Key: LongWritable (offset of line in file)
   * Input Value: Text (the line itself)
   * Output Key: Text (word)
   * Output Value: IntWritable (count = 1)
   */
  public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private final Text word = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
      // Case sensitive
      String line = value.toString();
      StringTokenizer tokenizer = new StringTokenizer(line);

      // Keep pooping (token, 1) pairs
      while (tokenizer.hasMoreTokens()) {
        value.set(tokenizer.nextToken());
        context.write(value, new IntWritable(1));
      }
    }
  }

  /**
   * Reducer class:
   * Input Key: Text (word)
   * Input Value: Iterable<IntWritable> (list of counts for that word)
   * Output Key: Text (word)
   * Output Value: IntWritable (sum of counts)
   */
  public static class SumReducer
      extends Reducer<Text, IntWritable, Text, IntWritable> {

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      // emit (word, total_count)
      context.write(key, new IntWritable(sum));
    }
  }

  /**
   * Driver method
   */
  @Override
  public int run(String[] args) throws Exception {
    
    Configuration conf = new Configuration();
    Job job = Job.getInstance(getConf(), "Yijia's word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setReducerClass(SumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    return job.waitForCompletion(true) ? 0 : 1;
  }

  public static void main(String[] args) throws Exception {
    int exitCode = ToolRunner.run(new WordCount(), args);
    System.exit(exitCode);
  }
}
