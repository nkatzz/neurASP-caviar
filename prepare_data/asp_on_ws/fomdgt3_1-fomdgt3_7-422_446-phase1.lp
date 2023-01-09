
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

	orientation(fomdgt3_1,150,422). coords(fomdgt3_1,46,102,422).  visible(fomdgt3_1,422). happensAt(inactive(fomdgt3_1),422). orientation(fomdgt3_7,150,422). coords(fomdgt3_7,81,116,422).  visible(fomdgt3_7,422). happensAt(walking(fomdgt3_7),422).
orientation(fomdgt3_1,150,423). coords(fomdgt3_1,46,102,423).  visible(fomdgt3_1,423). happensAt(inactive(fomdgt3_1),423). orientation(fomdgt3_7,150,423). coords(fomdgt3_7,82,117,423).  visible(fomdgt3_7,423). happensAt(walking(fomdgt3_7),423).
orientation(fomdgt3_1,150,424). coords(fomdgt3_1,46,102,424).  visible(fomdgt3_1,424). happensAt(inactive(fomdgt3_1),424). orientation(fomdgt3_7,150,424). coords(fomdgt3_7,82,117,424).  visible(fomdgt3_7,424). happensAt(walking(fomdgt3_7),424).
orientation(fomdgt3_1,150,425). coords(fomdgt3_1,46,102,425).  visible(fomdgt3_1,425). happensAt(inactive(fomdgt3_1),425). orientation(fomdgt3_7,150,425). coords(fomdgt3_7,84,119,425).  visible(fomdgt3_7,425). happensAt(walking(fomdgt3_7),425).
orientation(fomdgt3_1,150,426). coords(fomdgt3_1,46,102,426).  visible(fomdgt3_1,426). happensAt(inactive(fomdgt3_1),426). orientation(fomdgt3_7,150,426). coords(fomdgt3_7,84,121,426).  visible(fomdgt3_7,426). happensAt(walking(fomdgt3_7),426).
orientation(fomdgt3_1,150,427). coords(fomdgt3_1,46,102,427).  visible(fomdgt3_1,427). happensAt(inactive(fomdgt3_1),427). orientation(fomdgt3_7,150,427). coords(fomdgt3_7,84,122,427).  visible(fomdgt3_7,427). happensAt(walking(fomdgt3_7),427).
orientation(fomdgt3_1,150,428). coords(fomdgt3_1,46,102,428).  visible(fomdgt3_1,428). happensAt(inactive(fomdgt3_1),428). orientation(fomdgt3_7,150,428). coords(fomdgt3_7,85,123,428).  visible(fomdgt3_7,428). happensAt(walking(fomdgt3_7),428).
orientation(fomdgt3_1,150,429). coords(fomdgt3_1,46,102,429).  visible(fomdgt3_1,429). happensAt(inactive(fomdgt3_1),429). orientation(fomdgt3_7,150,429). coords(fomdgt3_7,85,124,429).  visible(fomdgt3_7,429). happensAt(walking(fomdgt3_7),429).
orientation(fomdgt3_1,150,430). coords(fomdgt3_1,46,102,430).  visible(fomdgt3_1,430). happensAt(inactive(fomdgt3_1),430). orientation(fomdgt3_7,150,430). coords(fomdgt3_7,86,124,430).  visible(fomdgt3_7,430). happensAt(walking(fomdgt3_7),430).
orientation(fomdgt3_1,150,431). coords(fomdgt3_1,46,102,431).  visible(fomdgt3_1,431). happensAt(inactive(fomdgt3_1),431). orientation(fomdgt3_7,150,431). coords(fomdgt3_7,86,124,431).  visible(fomdgt3_7,431). happensAt(walking(fomdgt3_7),431).
orientation(fomdgt3_1,150,432). coords(fomdgt3_1,46,102,432).  visible(fomdgt3_1,432). happensAt(inactive(fomdgt3_1),432). orientation(fomdgt3_7,150,432). coords(fomdgt3_7,86,125,432).  visible(fomdgt3_7,432). happensAt(walking(fomdgt3_7),432).
orientation(fomdgt3_1,150,433). coords(fomdgt3_1,46,102,433).  visible(fomdgt3_1,433). happensAt(inactive(fomdgt3_1),433). orientation(fomdgt3_7,150,433). coords(fomdgt3_7,86,125,433).  visible(fomdgt3_7,433). happensAt(walking(fomdgt3_7),433).
orientation(fomdgt3_1,150,434). coords(fomdgt3_1,46,102,434).  visible(fomdgt3_1,434). happensAt(inactive(fomdgt3_1),434). orientation(fomdgt3_7,150,434). coords(fomdgt3_7,86,126,434).  visible(fomdgt3_7,434). happensAt(walking(fomdgt3_7),434).
orientation(fomdgt3_1,150,435). coords(fomdgt3_1,46,102,435).  visible(fomdgt3_1,435). happensAt(inactive(fomdgt3_1),435). orientation(fomdgt3_7,150,435). coords(fomdgt3_7,86,126,435).  visible(fomdgt3_7,435). happensAt(walking(fomdgt3_7),435).
orientation(fomdgt3_1,150,436). coords(fomdgt3_1,46,102,436).  visible(fomdgt3_1,436). happensAt(inactive(fomdgt3_1),436). orientation(fomdgt3_7,150,436). coords(fomdgt3_7,86,126,436).  visible(fomdgt3_7,436). happensAt(walking(fomdgt3_7),436).
orientation(fomdgt3_1,150,437). coords(fomdgt3_1,46,102,437).  visible(fomdgt3_1,437). happensAt(inactive(fomdgt3_1),437). orientation(fomdgt3_7,150,437). coords(fomdgt3_7,86,127,437).  visible(fomdgt3_7,437). happensAt(walking(fomdgt3_7),437).
orientation(fomdgt3_1,150,438). coords(fomdgt3_1,46,102,438).  visible(fomdgt3_1,438). happensAt(inactive(fomdgt3_1),438). orientation(fomdgt3_7,150,438). coords(fomdgt3_7,86,129,438).  visible(fomdgt3_7,438). happensAt(walking(fomdgt3_7),438).
orientation(fomdgt3_1,150,439). coords(fomdgt3_1,46,102,439).  visible(fomdgt3_1,439). happensAt(inactive(fomdgt3_1),439). orientation(fomdgt3_7,150,439). coords(fomdgt3_7,86,131,439).  visible(fomdgt3_7,439). happensAt(walking(fomdgt3_7),439).
orientation(fomdgt3_1,150,440). coords(fomdgt3_1,46,102,440).  visible(fomdgt3_1,440). happensAt(inactive(fomdgt3_1),440). orientation(fomdgt3_7,150,440). coords(fomdgt3_7,86,131,440).  visible(fomdgt3_7,440). happensAt(walking(fomdgt3_7),440).
orientation(fomdgt3_1,150,441). coords(fomdgt3_1,46,102,441).  visible(fomdgt3_1,441). happensAt(inactive(fomdgt3_1),441). orientation(fomdgt3_7,150,441). coords(fomdgt3_7,88,133,441).  visible(fomdgt3_7,441). happensAt(walking(fomdgt3_7),441).
orientation(fomdgt3_1,150,442). coords(fomdgt3_1,46,102,442).  visible(fomdgt3_1,442). happensAt(inactive(fomdgt3_1),442). orientation(fomdgt3_7,150,442). coords(fomdgt3_7,88,133,442).  visible(fomdgt3_7,442). happensAt(walking(fomdgt3_7),442).
orientation(fomdgt3_1,150,443). coords(fomdgt3_1,46,102,443).  visible(fomdgt3_1,443). happensAt(inactive(fomdgt3_1),443). orientation(fomdgt3_7,150,443). coords(fomdgt3_7,89,134,443).  visible(fomdgt3_7,443). happensAt(walking(fomdgt3_7),443).
orientation(fomdgt3_1,150,444). coords(fomdgt3_1,46,102,444).  visible(fomdgt3_1,444). happensAt(inactive(fomdgt3_1),444). orientation(fomdgt3_7,150,444). coords(fomdgt3_7,90,135,444).  visible(fomdgt3_7,444). happensAt(walking(fomdgt3_7),444).
orientation(fomdgt3_1,150,445). coords(fomdgt3_1,46,102,445).  visible(fomdgt3_1,445). happensAt(inactive(fomdgt3_1),445). orientation(fomdgt3_7,150,445). coords(fomdgt3_7,91,136,445).  visible(fomdgt3_7,445). happensAt(walking(fomdgt3_7),445).
orientation(fomdgt3_1,150,446). coords(fomdgt3_1,46,102,446).  visible(fomdgt3_1,446). happensAt(inactive(fomdgt3_1),446). orientation(fomdgt3_7,150,446). coords(fomdgt3_7,91,136,446).  visible(fomdgt3_7,446). happensAt(walking(fomdgt3_7),446).
#show holdsAt/2.