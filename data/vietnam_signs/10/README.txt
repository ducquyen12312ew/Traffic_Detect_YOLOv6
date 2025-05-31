const mongoose = require('mongoose');
const connect = mongoose.connect("mongodb://0.0.0.0:27017/Pet");

connect.then(() => {
    console.log("Database Connected Successfully");
})
.catch(() => {
    console.log("Database cannot be Connected");
});
const UserSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    password: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true
    },
    phone: {
        type: String
    },
    role: {
        type: String,
        enum: ['admin', 'vet', 'staff', 'user'], 
        default: 'user'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

const PetSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    type: {
        type: String,
        required: true
    },
    breed: {
        type: String
    },
    age: {
        type: Number
    },
    weight: {
        type: Number
    },
    gender: {
        type: String,
        enum: ['male', 'female', 'unknown'],
        default: 'unknown'
    },
    ownerName: {
        type: String,
        required: true
    },
    ownerPhone: {
        type: String,
        required: true
    },
    ownerEmail: {
        type: String
    },
    registeredAt: {
        type: Date,
        default: Date.now
    }
});

const HealthRecordSchema = new mongoose.Schema({
    pet: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'pets',
        required: true
    },
    date: {
        type: Date,
        default: Date.now
    },
    veterinarian: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'users'
    },
    temperature: Number,
    weight: Number,
    heartRate: Number,
    respirationRate: Number,
    notes: String,
    nextCheckupDate: Date,
    status: {
        type: String,
        enum: ['healthy', 'sick', 'recovering', 'critical'],
        default: 'healthy'
    }
});

const MedicalRecordSchema = new mongoose.Schema({
    pet: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'pets',
        required: true
    },
    date: {
        type: Date,
        default: Date.now
    },
    veterinarian: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'users'
    },
    symptoms: [String],
    diagnosis: String,
    treatment: String,
    notes: String,
    followUpDate: Date
});

const PrescriptionSchema = new mongoose.Schema({
    pet: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'pets',
        required: true
    },
    date: {
        type: Date,
        default: Date.now
    },
    veterinarian: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'users'
    },
    medications: [{
        name: String,
        dosage: String,
        frequency: String,
        duration: String,
        notes: String
    }],
    instructions: String,
    status: {
        type: String,
        enum: ['active', 'completed', 'cancelled'],
        default: 'active'
    },

    medicalRecord: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'medicalRecords'
    }
});

const AppointmentSchema = new mongoose.Schema({
    customerName: {
        type: String,
        required: true
    },
    customerEmail: {
        type: String,
        required: true
    },
    customerPhone: {
        type: String,
        required: true
    },
    petName: {
        type: String,
        required: true
    },
    petType: {
        type: String,
        required: true
    },
    petBreed: {
        type: String
    },
    service: {
        type: String,
        required: true,
        enum: ['khám sức khỏe', 'tắm', 'cắt tỉa lông', 'lưu trú']
    },
    date: {
        type: Date,
        required: true
    },
    time: {
        type: String,
        required: true
    },
    notes: {
        type: String
    },
    status: {
        type: String,
        enum: ['pending', 'confirmed', 'completed', 'cancelled'],
        default: 'pending'
    },
    assignedVet: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'users'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});
const ShopInformationSchema = new mongoose.Schema({
    shopName: {
        type: String,
        required: true,
        default: "Pet Care Center"
    },
    address: {
        type: String,
        required: true
    },
    phone: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true
    },
    description: {
        type: String
    },
    workingHours: {
        monday: { open: String, close: String },
        tuesday: { open: String, close: String },
        wednesday: { open: String, close: String },
        thursday: { open: String, close: String },
        friday: { open: String, close: String },
        saturday: { open: String, close: String },
        sunday: { open: String, close: String }
    },
    services: [String],
    socialMedia: {
        facebook: String,
        instagram: String,
        website: String
    },
    updatedAt: {
        type: Date,
        default: Date.now
    },
    updatedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'users'
    }
});

const ShopInformationCollection = mongoose.model("shopInformation", ShopInformationSchema);
const UserCollection = mongoose.model("users", UserSchema);
const PetCollection = mongoose.model("pets", PetSchema);
const HealthRecordCollection = mongoose.model("healthRecords", HealthRecordSchema);
const MedicalRecordCollection = mongoose.model("medicalRecords", MedicalRecordSchema);
const PrescriptionCollection = mongoose.model("prescriptions", PrescriptionSchema);
const AppointmentCollection = mongoose.model("appointments", AppointmentSchema);

module.exports = { 
    UserCollection, 
    PetCollection, 
    HealthRecordCollection, 
    MedicalRecordCollection, 
    PrescriptionCollection, 
    AppointmentCollection,
    ShopInformationCollection 
};