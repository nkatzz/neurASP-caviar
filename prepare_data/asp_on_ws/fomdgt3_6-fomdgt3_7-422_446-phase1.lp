
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

	orientation(fomdgt3_6,140,422). coords(fomdgt3_6,59,89,422).  visible(fomdgt3_6,422). happensAt(walking(fomdgt3_6),422). orientation(fomdgt3_7,150,422). coords(fomdgt3_7,81,116,422).  visible(fomdgt3_7,422). happensAt(walking(fomdgt3_7),422).
orientation(fomdgt3_6,140,423). coords(fomdgt3_6,58,89,423).  visible(fomdgt3_6,423). happensAt(walking(fomdgt3_6),423). orientation(fomdgt3_7,150,423). coords(fomdgt3_7,82,117,423).  visible(fomdgt3_7,423). happensAt(walking(fomdgt3_7),423).
orientation(fomdgt3_6,160,424). coords(fomdgt3_6,54,90,424).  visible(fomdgt3_6,424). happensAt(walking(fomdgt3_6),424). orientation(fomdgt3_7,150,424). coords(fomdgt3_7,82,117,424).  visible(fomdgt3_7,424). happensAt(walking(fomdgt3_7),424).
orientation(fomdgt3_6,160,425). coords(fomdgt3_6,53,91,425).  visible(fomdgt3_6,425). happensAt(walking(fomdgt3_6),425). orientation(fomdgt3_7,150,425). coords(fomdgt3_7,84,119,425).  visible(fomdgt3_7,425). happensAt(walking(fomdgt3_7),425).
orientation(fomdgt3_6,160,426). coords(fomdgt3_6,53,91,426).  visible(fomdgt3_6,426). happensAt(walking(fomdgt3_6),426). orientation(fomdgt3_7,150,426). coords(fomdgt3_7,84,121,426).  visible(fomdgt3_7,426). happensAt(walking(fomdgt3_7),426).
orientation(fomdgt3_6,160,427). coords(fomdgt3_6,51,91,427).  visible(fomdgt3_6,427). happensAt(walking(fomdgt3_6),427). orientation(fomdgt3_7,150,427). coords(fomdgt3_7,84,122,427).  visible(fomdgt3_7,427). happensAt(walking(fomdgt3_7),427).
orientation(fomdgt3_6,160,428). coords(fomdgt3_6,51,92,428).  visible(fomdgt3_6,428). happensAt(walking(fomdgt3_6),428). orientation(fomdgt3_7,150,428). coords(fomdgt3_7,85,123,428).  visible(fomdgt3_7,428). happensAt(walking(fomdgt3_7),428).
orientation(fomdgt3_6,160,429). coords(fomdgt3_6,51,93,429).  visible(fomdgt3_6,429). happensAt(walking(fomdgt3_6),429). orientation(fomdgt3_7,150,429). coords(fomdgt3_7,85,124,429).  visible(fomdgt3_7,429). happensAt(walking(fomdgt3_7),429).
orientation(fomdgt3_6,160,430). coords(fomdgt3_6,51,93,430).  visible(fomdgt3_6,430). happensAt(walking(fomdgt3_6),430). orientation(fomdgt3_7,150,430). coords(fomdgt3_7,86,124,430).  visible(fomdgt3_7,430). happensAt(walking(fomdgt3_7),430).
orientation(fomdgt3_6,160,431). coords(fomdgt3_6,50,93,431).  visible(fomdgt3_6,431). happensAt(walking(fomdgt3_6),431). orientation(fomdgt3_7,150,431). coords(fomdgt3_7,86,124,431).  visible(fomdgt3_7,431). happensAt(walking(fomdgt3_7),431).
orientation(fomdgt3_6,160,432). coords(fomdgt3_6,49,93,432).  visible(fomdgt3_6,432). happensAt(walking(fomdgt3_6),432). orientation(fomdgt3_7,150,432). coords(fomdgt3_7,86,125,432).  visible(fomdgt3_7,432). happensAt(walking(fomdgt3_7),432).
orientation(fomdgt3_6,160,433). coords(fomdgt3_6,49,94,433).  visible(fomdgt3_6,433). happensAt(walking(fomdgt3_6),433). orientation(fomdgt3_7,150,433). coords(fomdgt3_7,86,125,433).  visible(fomdgt3_7,433). happensAt(walking(fomdgt3_7),433).
orientation(fomdgt3_6,160,434). coords(fomdgt3_6,49,94,434).  visible(fomdgt3_6,434). happensAt(walking(fomdgt3_6),434). orientation(fomdgt3_7,150,434). coords(fomdgt3_7,86,126,434).  visible(fomdgt3_7,434). happensAt(walking(fomdgt3_7),434).
orientation(fomdgt3_6,160,435). coords(fomdgt3_6,49,94,435).  visible(fomdgt3_6,435). happensAt(walking(fomdgt3_6),435). orientation(fomdgt3_7,150,435). coords(fomdgt3_7,86,126,435).  visible(fomdgt3_7,435). happensAt(walking(fomdgt3_7),435).
orientation(fomdgt3_6,160,436). coords(fomdgt3_6,49,95,436).  visible(fomdgt3_6,436). happensAt(walking(fomdgt3_6),436). orientation(fomdgt3_7,150,436). coords(fomdgt3_7,86,126,436).  visible(fomdgt3_7,436). happensAt(walking(fomdgt3_7),436).
orientation(fomdgt3_6,160,437). coords(fomdgt3_6,49,97,437).  visible(fomdgt3_6,437). happensAt(walking(fomdgt3_6),437). orientation(fomdgt3_7,150,437). coords(fomdgt3_7,86,127,437).  visible(fomdgt3_7,437). happensAt(walking(fomdgt3_7),437).
orientation(fomdgt3_6,160,438). coords(fomdgt3_6,49,97,438).  visible(fomdgt3_6,438). happensAt(walking(fomdgt3_6),438). orientation(fomdgt3_7,150,438). coords(fomdgt3_7,86,129,438).  visible(fomdgt3_7,438). happensAt(walking(fomdgt3_7),438).
orientation(fomdgt3_6,160,439). coords(fomdgt3_6,49,103,439).  visible(fomdgt3_6,439). happensAt(walking(fomdgt3_6),439). orientation(fomdgt3_7,150,439). coords(fomdgt3_7,86,131,439).  visible(fomdgt3_7,439). happensAt(walking(fomdgt3_7),439).
orientation(fomdgt3_6,160,440). coords(fomdgt3_6,49,103,440).  visible(fomdgt3_6,440). happensAt(walking(fomdgt3_6),440). orientation(fomdgt3_7,150,440). coords(fomdgt3_7,86,131,440).  visible(fomdgt3_7,440). happensAt(walking(fomdgt3_7),440).
orientation(fomdgt3_6,160,441). coords(fomdgt3_6,49,103,441).  visible(fomdgt3_6,441). happensAt(walking(fomdgt3_6),441). orientation(fomdgt3_7,150,441). coords(fomdgt3_7,88,133,441).  visible(fomdgt3_7,441). happensAt(walking(fomdgt3_7),441).
orientation(fomdgt3_6,160,442). coords(fomdgt3_6,49,103,442).  visible(fomdgt3_6,442). happensAt(walking(fomdgt3_6),442). orientation(fomdgt3_7,150,442). coords(fomdgt3_7,88,133,442).  visible(fomdgt3_7,442). happensAt(walking(fomdgt3_7),442).
orientation(fomdgt3_6,160,443). coords(fomdgt3_6,49,103,443).  visible(fomdgt3_6,443). happensAt(walking(fomdgt3_6),443). orientation(fomdgt3_7,150,443). coords(fomdgt3_7,89,134,443).  visible(fomdgt3_7,443). happensAt(walking(fomdgt3_7),443).
orientation(fomdgt3_6,160,444). coords(fomdgt3_6,49,103,444).  visible(fomdgt3_6,444). happensAt(walking(fomdgt3_6),444). orientation(fomdgt3_7,150,444). coords(fomdgt3_7,90,135,444).  visible(fomdgt3_7,444). happensAt(walking(fomdgt3_7),444).
orientation(fomdgt3_6,160,445). coords(fomdgt3_6,49,103,445).  visible(fomdgt3_6,445). happensAt(walking(fomdgt3_6),445). orientation(fomdgt3_7,150,445). coords(fomdgt3_7,91,136,445).  visible(fomdgt3_7,445). happensAt(walking(fomdgt3_7),445).
orientation(fomdgt3_6,160,446). coords(fomdgt3_6,49,103,446).  visible(fomdgt3_6,446). happensAt(walking(fomdgt3_6),446). orientation(fomdgt3_7,150,446). coords(fomdgt3_7,91,136,446).  visible(fomdgt3_7,446). happensAt(walking(fomdgt3_7),446).
#show holdsAt/2.