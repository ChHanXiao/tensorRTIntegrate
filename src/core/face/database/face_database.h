#ifndef FACE_DATABASE_H
#define FACE_DATABASE_H

#include <cc_util.hpp>
#include "./stream/file_stream.h"

#define kFaceFeatureDim 512
#define kFaceNameDim 256

class FaceDatabase {
public:
	FaceDatabase();
	~FaceDatabase();

	bool Save(const char* path) const;
	bool Load(const char* path);
	int64_t Insert(const std::vector<float>& feat, const std::string& name);
	int Delete(const std::string& name);
	int64_t QueryTop(const std::vector<float>& feat, ccutil::QueryResult* query_result = nullptr);
	bool Clear();

private:
	FaceDatabase(const FaceDatabase &other) = delete;
	const FaceDatabase &operator=(const FaceDatabase &other) = delete;

private:
	class Impl;
	Impl* impl_;

};


#endif // !FACE_DATABASE_H

