#!/bin/sh

# specifying dependencies - change paths accordingly if needed
# all OpenIE dependencies should lie in ivy cache
OPENIE_DEPS=`find ~/.ivy2/cache/ -name '*.jar' -exec echo {} \; | tr '\n' ':' | sed 's/:$/\n/'`
OPENIE="../../../openie-standalone/target/scala-2.11/classes"
JSON_DEPS="deps/json4s-core_2.11-3.4.1.jar:deps/json4s-native_2.11-3.4.1.jar:deps/json4s-ast_2.10-3.4.1.jar"

exec env JAVA_OPTS="-Xmx4G -XX:+UseConcMarkSweepGC" \
    scala -classpath .:$JSON_DEPS:$OPENIE:$OPENIE_DEPS \
    extractor.scala

