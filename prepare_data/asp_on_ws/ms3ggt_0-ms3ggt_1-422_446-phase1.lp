
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(ms3ggt_0,157,422). coords(ms3ggt_0,121,145,422).  visible(ms3ggt_0,422). happensAt(walking(ms3ggt_0),422). orientation(ms3ggt_1,163,422). coords(ms3ggt_1,172,158,422).  visible(ms3ggt_1,422). happensAt(walking(ms3ggt_1),422).
orientation(ms3ggt_0,157,423). coords(ms3ggt_0,121,145,423).  visible(ms3ggt_0,423). happensAt(walking(ms3ggt_0),423). orientation(ms3ggt_1,163,423). coords(ms3ggt_1,172,157,423).  visible(ms3ggt_1,423). happensAt(walking(ms3ggt_1),423).
orientation(ms3ggt_0,157,424). coords(ms3ggt_0,120,145,424).  visible(ms3ggt_0,424). happensAt(walking(ms3ggt_0),424). orientation(ms3ggt_1,163,424). coords(ms3ggt_1,173,157,424).  visible(ms3ggt_1,424). happensAt(walking(ms3ggt_1),424).
orientation(ms3ggt_0,157,425). coords(ms3ggt_0,120,145,425).  visible(ms3ggt_0,425). happensAt(walking(ms3ggt_0),425). orientation(ms3ggt_1,163,425). coords(ms3ggt_1,173,157,425).  visible(ms3ggt_1,425). happensAt(walking(ms3ggt_1),425).
orientation(ms3ggt_0,157,426). coords(ms3ggt_0,119,145,426).  visible(ms3ggt_0,426). happensAt(walking(ms3ggt_0),426). orientation(ms3ggt_1,163,426). coords(ms3ggt_1,174,156,426).  visible(ms3ggt_1,426). happensAt(walking(ms3ggt_1),426).
orientation(ms3ggt_0,157,427). coords(ms3ggt_0,119,145,427).  visible(ms3ggt_0,427). happensAt(walking(ms3ggt_0),427). orientation(ms3ggt_1,163,427). coords(ms3ggt_1,174,156,427).  visible(ms3ggt_1,427). happensAt(walking(ms3ggt_1),427).
orientation(ms3ggt_0,157,428). coords(ms3ggt_0,118,145,428).  visible(ms3ggt_0,428). happensAt(walking(ms3ggt_0),428). orientation(ms3ggt_1,154,428). coords(ms3ggt_1,175,157,428).  visible(ms3ggt_1,428). happensAt(walking(ms3ggt_1),428).
orientation(ms3ggt_0,157,429). coords(ms3ggt_0,118,145,429).  visible(ms3ggt_0,429). happensAt(walking(ms3ggt_0),429). orientation(ms3ggt_1,154,429). coords(ms3ggt_1,175,157,429).  visible(ms3ggt_1,429). happensAt(walking(ms3ggt_1),429).
orientation(ms3ggt_0,157,430). coords(ms3ggt_0,116,145,430).  visible(ms3ggt_0,430). happensAt(walking(ms3ggt_0),430). orientation(ms3ggt_1,154,430). coords(ms3ggt_1,176,157,430).  visible(ms3ggt_1,430). happensAt(walking(ms3ggt_1),430).
orientation(ms3ggt_0,160,431). coords(ms3ggt_0,114,144,431).  visible(ms3ggt_0,431). happensAt(walking(ms3ggt_0),431). orientation(ms3ggt_1,158,431). coords(ms3ggt_1,181,159,431).  visible(ms3ggt_1,431). happensAt(walking(ms3ggt_1),431).
orientation(ms3ggt_0,160,432). coords(ms3ggt_0,112,144,432).  visible(ms3ggt_0,432). happensAt(walking(ms3ggt_0),432). orientation(ms3ggt_1,158,432). coords(ms3ggt_1,184,159,432).  visible(ms3ggt_1,432). happensAt(walking(ms3ggt_1),432).
orientation(ms3ggt_0,160,433). coords(ms3ggt_0,111,144,433).  visible(ms3ggt_0,433). happensAt(walking(ms3ggt_0),433). orientation(ms3ggt_1,158,433). coords(ms3ggt_1,186,161,433).  visible(ms3ggt_1,433). happensAt(walking(ms3ggt_1),433).
orientation(ms3ggt_0,160,434). coords(ms3ggt_0,109,144,434).  visible(ms3ggt_0,434). happensAt(walking(ms3ggt_0),434). orientation(ms3ggt_1,158,434). coords(ms3ggt_1,188,161,434).  visible(ms3ggt_1,434). happensAt(walking(ms3ggt_1),434).
orientation(ms3ggt_0,160,435). coords(ms3ggt_0,108,144,435).  visible(ms3ggt_0,435). happensAt(walking(ms3ggt_0),435). orientation(ms3ggt_1,158,435). coords(ms3ggt_1,188,161,435).  visible(ms3ggt_1,435). happensAt(walking(ms3ggt_1),435).
orientation(ms3ggt_0,160,436). coords(ms3ggt_0,107,145,436).  visible(ms3ggt_0,436). happensAt(walking(ms3ggt_0),436). orientation(ms3ggt_1,158,436). coords(ms3ggt_1,189,161,436).  visible(ms3ggt_1,436). happensAt(walking(ms3ggt_1),436).
orientation(ms3ggt_0,160,437). coords(ms3ggt_0,107,146,437).  visible(ms3ggt_0,437). happensAt(walking(ms3ggt_0),437). orientation(ms3ggt_1,158,437). coords(ms3ggt_1,189,161,437).  visible(ms3ggt_1,437). happensAt(walking(ms3ggt_1),437).
orientation(ms3ggt_0,160,438). coords(ms3ggt_0,107,146,438).  visible(ms3ggt_0,438). happensAt(walking(ms3ggt_0),438). orientation(ms3ggt_1,158,438). coords(ms3ggt_1,191,161,438).  visible(ms3ggt_1,438). happensAt(walking(ms3ggt_1),438).
orientation(ms3ggt_0,160,439). coords(ms3ggt_0,107,146,439).  visible(ms3ggt_0,439). happensAt(walking(ms3ggt_0),439). orientation(ms3ggt_1,158,439). coords(ms3ggt_1,191,161,439).  visible(ms3ggt_1,439). happensAt(walking(ms3ggt_1),439).
orientation(ms3ggt_0,168,440). coords(ms3ggt_0,106,146,440).  visible(ms3ggt_0,440). happensAt(walking(ms3ggt_0),440). orientation(ms3ggt_1,158,440). coords(ms3ggt_1,192,161,440).  visible(ms3ggt_1,440). happensAt(walking(ms3ggt_1),440).
orientation(ms3ggt_0,168,441). coords(ms3ggt_0,105,147,441).  visible(ms3ggt_0,441). happensAt(walking(ms3ggt_0),441). orientation(ms3ggt_1,158,441). coords(ms3ggt_1,193,163,441).  visible(ms3ggt_1,441). happensAt(walking(ms3ggt_1),441).
orientation(ms3ggt_0,168,442). coords(ms3ggt_0,105,147,442).  visible(ms3ggt_0,442). happensAt(walking(ms3ggt_0),442). orientation(ms3ggt_1,167,442). coords(ms3ggt_1,196,164,442).  visible(ms3ggt_1,442). happensAt(walking(ms3ggt_1),442).
orientation(ms3ggt_0,170,443). coords(ms3ggt_0,103,147,443).  visible(ms3ggt_0,443). happensAt(walking(ms3ggt_0),443). orientation(ms3ggt_1,171,443). coords(ms3ggt_1,199,165,443).  visible(ms3ggt_1,443). happensAt(walking(ms3ggt_1),443).
orientation(ms3ggt_0,171,444). coords(ms3ggt_0,101,148,444).  visible(ms3ggt_0,444). happensAt(walking(ms3ggt_0),444). orientation(ms3ggt_1,171,444). coords(ms3ggt_1,202,165,444).  visible(ms3ggt_1,444). happensAt(walking(ms3ggt_1),444).
orientation(ms3ggt_0,171,445). coords(ms3ggt_0,100,148,445).  visible(ms3ggt_0,445). happensAt(walking(ms3ggt_0),445). orientation(ms3ggt_1,177,445). coords(ms3ggt_1,206,166,445).  visible(ms3ggt_1,445). happensAt(walking(ms3ggt_1),445).
orientation(ms3ggt_0,171,446). coords(ms3ggt_0,98,148,446).  visible(ms3ggt_0,446). happensAt(walking(ms3ggt_0),446). orientation(ms3ggt_1,177,446). coords(ms3ggt_1,208,166,446).  visible(ms3ggt_1,446). happensAt(walking(ms3ggt_1),446).
#show holdsAt/2.