import { z } from "zod";

// Test enum
const MyEnum = z.enum(["a", "b", "c"]);
console.log("MyEnum.def:", MyEnum.def);
console.log("MyEnum.def.type:", MyEnum.def.type);
console.log("MyEnum.def.entries:", MyEnum.def.entries);
console.log("MyEnum.description:", MyEnum.description);

// Test string
const MyString = z.string().describe("A string field");
console.log("\nMyString.def:", MyString.def);
console.log("MyString.def.type:", MyString.def.type);
console.log("MyString.description:", MyString.description);

// Test object
const MyObject = z.object({
  name: z.string().describe("User name"),
  email: z.string().email(),
  role: z.enum(["admin", "user"]).describe("User role"),
});

console.log("\nMyObject.shape:", MyObject.shape);
for (const [key, val] of Object.entries(MyObject.shape)) {
  console.log(`\n${key}:`, {
    def: (val as any).def,
    defType: (val as any).def?.type,
    description: (val as any).description,
  });
}
