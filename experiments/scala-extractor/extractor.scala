import edu.knowitall.openie.OpenIE
import edu.knowitall.openie.OpenIECli.{ColumnFormat, SimpleFormat}
import edu.knowitall.openie.Instance

import scala.io.Source
import java.io._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.JsonDSL._


object Extractor {
    val openie = new OpenIE()

    // Invoke method that you need
    def main(args: Array[String]) = {
        // assuming each line is the separate sentence
        val input_path = "../samples.txt"
        val output_path = "../samples-relations.txt"

        //extract_to_stdout(input_path, output_path)
        extract_to_file(input_path, output_path)
        //extract_to_json(input_path, output_path)
    }

    def extract_to_stdout(input_path: String, output_path: String) {
        val sw = new StringWriter()
        val pw = new PrintWriter(sw)

        for (line <- Source.fromFile(input_path).getLines()) {
            val instances = openie.extract(line)

            SimpleFormat.print(pw, line, instances)
            println(sw.toString())
        }
    }

    def extract_to_file(input_path: String, output_path: String) {
        println("EXTRACTING TO FILE")
        
        val file = new File(output_path)
        val bw = new BufferedWriter(new FileWriter(file))
        val pw = new PrintWriter(bw)

        for (line <- Source.fromFile(input_path).getLines()) {
            val instances = openie.extract(line)

            //SimpleFormat.print(pw, line, instances)
            ColumnFormat.print(pw, line, instances)
            bw.newLine()
        }
        bw.close()
    }

    def extract_to_json(input_path: String, output_path: String) {
        println("EXTRACTING TO JSON")

        val file = new File(output_path)
        val bw = new BufferedWriter(new FileWriter(file))
        val pw = new PrintWriter(bw)

        def get_context(inst: Instance, def_value: String  = "") = {
            inst.extr.context match {
              case Some(context) => context.displayText
              case None => def_value
            }
        }

        def gen_json_for_insts(instances: Seq[Instance]) = 
          for (inst <- instances) yield
               ("confidence" -> inst.confidence) ~ 
               ("extraction" -> List(
                 ("context" -> get_context(inst)) ~
                 ("arg1" -> inst.extr.arg1.displayText) ~
                 ("rel" -> inst.extr.rel.displayText) ~
                 ("arg2s" -> inst.extr.arg2s.map(_.displayText).mkString("; "))))
            
        for (line <- Source.fromFile(input_path).getLines()) {
            val instances = openie.extract(line)
            val sentence_json = ("sentence" -> line) ~ ("instances" -> gen_json_for_insts(instances))

            // output to stdout for user
            println(pretty(render(sentence_json)))
            bw.write((compact(render(sentence_json))))
            bw.newLine()
        }
        bw.close()
    }
}

