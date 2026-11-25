/**
 * K6.8a / K6.8b: reading a core uniform struct from a buffer three binds as storage. three r187 binds a buffer it does not
 * own only as storage (`vdbThreeStorageStandIn`, the `VDBGridNode` pattern): a uniform `BufferNode` uploads its own array.
 * So the core buffer (a volume shadow map's `view`, the emission probes' `grid`) is read as `array<vec4u>` and a generated
 * WGSL function unpacks it into the core's struct, field by field from the struct's layout, so a field the core adds is
 * unpacked too. The struct's own WGSL comes with the core's receiver code (`vdbVolumeShadowWgsl()`, `vdbEmissionProbeWgsl()`).
 */

import { CodeNode, type Node, type NodeBuilder } from "three/webgpu";
import type { VDBUniformStruct } from "../../core/frame/uniforms.js";

/** A three storage node (`storage(attribute, 'uvec4', words).toReadOnly()`). */
export type VDBFrameStorageNode = Node & { value: unknown; setName(name: string): unknown; build(builder: NodeBuilder, output?: string): unknown };

/**
 * WGSL `fn <fnName>() -> <struct.name>` that unpacks the struct from the `array<vec4u>` binding `ref`. Only 16-byte vector
 * and `mat4x4f` fields unpack (every field of the structs three reads); another type throws.
 */
export function vdbFrameStructLoaderWgsl(struct: VDBUniformStruct, fnName: string, ref: string): string {
  const lines = [`fn ${fnName}() -> ${struct.name} {`, `  var v: ${struct.name};`];
  for (const f of struct.fields.values()) {
    for (let e = 0; e < f.count; e++) {
      const at = (f.offset + e * f.stride) / 16;
      const lhs = f.count > 1 ? `v.${f.name}[${e}]` : `v.${f.name}`;
      let rhs: string;
      if (f.type === "mat4x4f") rhs = `mat4x4f(${[0, 1, 2, 3].map((c) => `bitcast<vec4f>(${ref}[${at + c}])`).join(", ")})`;
      else if (f.type === "vec4f") rhs = `bitcast<vec4f>(${ref}[${at}])`;
      else if (f.type === "vec4u") rhs = `${ref}[${at}]`;
      else throw new Error(`vdbFrameStructLoaderWgsl: ${struct.name}.${f.name} is ${f.type}; only 16-byte vector and mat4x4f fields unpack`);
      lines.push(`  ${lhs} = ${rhs};`);
    }
  }
  lines.push("  return v;", "}");
  return lines.join("\n");
}

/** `vec4u` words of a struct (its size is a multiple of 16). */
export const vdbFrameStructWords = (struct: VDBUniformStruct): number => struct.size / 16;

/**
 * One binding's loader: a code node over its storage node, generated per shader with the binding's reference (three names
 * the binding per shader). Include it in the `wgslFn` that calls `fnName()`.
 */
export class VDBFrameStructLoaderNode extends CodeNode {
  /** Node-valued property: three sets it up and analyzes it as this node's child (the binding). */
  vdbStructBuffer: VDBFrameStorageNode;
  readonly fnName: string;
  readonly struct: VDBUniformStruct;

  constructor(buffer: VDBFrameStorageNode, struct: VDBUniformStruct, fnName: string) {
    super("", [], "wgsl");
    this.vdbStructBuffer = buffer;
    this.struct = struct;
    this.fnName = fnName;
  }

  override generate(builder: NodeBuilder): string {
    const ref = this.vdbStructBuffer.build(builder, "property");
    if (typeof ref !== "string") throw new Error(`VDBFrameStructLoaderNode: three built the ${this.struct.name} storage node to a non-string reference`);
    const text = vdbFrameStructLoaderWgsl(this.struct, this.fnName, ref);
    const nodeCode = (builder as unknown as { getCodeFromNode(node: Node, type: string): { code: string } }).getCodeFromNode(this, this.getNodeType(builder));
    nodeCode.code = text;
    return text;
  }
}
