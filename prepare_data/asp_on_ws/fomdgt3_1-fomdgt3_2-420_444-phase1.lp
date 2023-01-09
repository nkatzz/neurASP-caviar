
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

	orientation(fomdgt3_1,150,420). coords(fomdgt3_1,46,102,420).  visible(fomdgt3_1,420). happensAt(inactive(fomdgt3_1),420). orientation(fomdgt3_2,42,420). coords(fomdgt3_2,262,96,420).  visible(fomdgt3_2,420). happensAt(inactive(fomdgt3_2),420).
orientation(fomdgt3_1,150,421). coords(fomdgt3_1,46,102,421).  visible(fomdgt3_1,421). happensAt(inactive(fomdgt3_1),421). orientation(fomdgt3_2,42,421). coords(fomdgt3_2,262,96,421).  visible(fomdgt3_2,421). happensAt(inactive(fomdgt3_2),421).
orientation(fomdgt3_1,150,422). coords(fomdgt3_1,46,102,422).  visible(fomdgt3_1,422). happensAt(inactive(fomdgt3_1),422). orientation(fomdgt3_2,42,422). coords(fomdgt3_2,262,96,422).  visible(fomdgt3_2,422). happensAt(inactive(fomdgt3_2),422).
orientation(fomdgt3_1,150,423). coords(fomdgt3_1,46,102,423).  visible(fomdgt3_1,423). happensAt(inactive(fomdgt3_1),423). orientation(fomdgt3_2,42,423). coords(fomdgt3_2,262,96,423).  visible(fomdgt3_2,423). happensAt(inactive(fomdgt3_2),423).
orientation(fomdgt3_1,150,424). coords(fomdgt3_1,46,102,424).  visible(fomdgt3_1,424). happensAt(inactive(fomdgt3_1),424). orientation(fomdgt3_2,42,424). coords(fomdgt3_2,262,96,424).  visible(fomdgt3_2,424). happensAt(inactive(fomdgt3_2),424).
orientation(fomdgt3_1,150,425). coords(fomdgt3_1,46,102,425).  visible(fomdgt3_1,425). happensAt(inactive(fomdgt3_1),425). orientation(fomdgt3_2,42,425). coords(fomdgt3_2,262,96,425).  visible(fomdgt3_2,425). happensAt(inactive(fomdgt3_2),425).
orientation(fomdgt3_1,150,426). coords(fomdgt3_1,46,102,426).  visible(fomdgt3_1,426). happensAt(inactive(fomdgt3_1),426). orientation(fomdgt3_2,42,426). coords(fomdgt3_2,262,96,426).  visible(fomdgt3_2,426). happensAt(inactive(fomdgt3_2),426).
orientation(fomdgt3_1,150,427). coords(fomdgt3_1,46,102,427).  visible(fomdgt3_1,427). happensAt(inactive(fomdgt3_1),427). orientation(fomdgt3_2,42,427). coords(fomdgt3_2,262,96,427).  visible(fomdgt3_2,427). happensAt(inactive(fomdgt3_2),427).
orientation(fomdgt3_1,150,428). coords(fomdgt3_1,46,102,428).  visible(fomdgt3_1,428). happensAt(inactive(fomdgt3_1),428). orientation(fomdgt3_2,42,428). coords(fomdgt3_2,262,96,428).  visible(fomdgt3_2,428). happensAt(inactive(fomdgt3_2),428).
orientation(fomdgt3_1,150,429). coords(fomdgt3_1,46,102,429).  visible(fomdgt3_1,429). happensAt(inactive(fomdgt3_1),429). orientation(fomdgt3_2,42,429). coords(fomdgt3_2,262,96,429).  visible(fomdgt3_2,429). happensAt(inactive(fomdgt3_2),429).
orientation(fomdgt3_1,150,430). coords(fomdgt3_1,46,102,430).  visible(fomdgt3_1,430). happensAt(inactive(fomdgt3_1),430). orientation(fomdgt3_2,42,430). coords(fomdgt3_2,262,96,430).  visible(fomdgt3_2,430). happensAt(inactive(fomdgt3_2),430).
orientation(fomdgt3_1,150,431). coords(fomdgt3_1,46,102,431).  visible(fomdgt3_1,431). happensAt(inactive(fomdgt3_1),431). orientation(fomdgt3_2,42,431). coords(fomdgt3_2,262,96,431).  visible(fomdgt3_2,431). happensAt(inactive(fomdgt3_2),431).
orientation(fomdgt3_1,150,432). coords(fomdgt3_1,46,102,432).  visible(fomdgt3_1,432). happensAt(inactive(fomdgt3_1),432). orientation(fomdgt3_2,42,432). coords(fomdgt3_2,262,96,432).  visible(fomdgt3_2,432). happensAt(inactive(fomdgt3_2),432).
orientation(fomdgt3_1,150,433). coords(fomdgt3_1,46,102,433).  visible(fomdgt3_1,433). happensAt(inactive(fomdgt3_1),433). orientation(fomdgt3_2,42,433). coords(fomdgt3_2,262,96,433).  visible(fomdgt3_2,433). happensAt(inactive(fomdgt3_2),433).
orientation(fomdgt3_1,150,434). coords(fomdgt3_1,46,102,434).  visible(fomdgt3_1,434). happensAt(inactive(fomdgt3_1),434). orientation(fomdgt3_2,42,434). coords(fomdgt3_2,262,96,434).  visible(fomdgt3_2,434). happensAt(inactive(fomdgt3_2),434).
orientation(fomdgt3_1,150,435). coords(fomdgt3_1,46,102,435).  visible(fomdgt3_1,435). happensAt(inactive(fomdgt3_1),435). orientation(fomdgt3_2,42,435). coords(fomdgt3_2,262,96,435).  visible(fomdgt3_2,435). happensAt(inactive(fomdgt3_2),435).
orientation(fomdgt3_1,150,436). coords(fomdgt3_1,46,102,436).  visible(fomdgt3_1,436). happensAt(inactive(fomdgt3_1),436). orientation(fomdgt3_2,42,436). coords(fomdgt3_2,262,96,436).  visible(fomdgt3_2,436). happensAt(inactive(fomdgt3_2),436).
orientation(fomdgt3_1,150,437). coords(fomdgt3_1,46,102,437).  visible(fomdgt3_1,437). happensAt(inactive(fomdgt3_1),437). orientation(fomdgt3_2,42,437). coords(fomdgt3_2,262,96,437).  visible(fomdgt3_2,437). happensAt(inactive(fomdgt3_2),437).
orientation(fomdgt3_1,150,438). coords(fomdgt3_1,46,102,438).  visible(fomdgt3_1,438). happensAt(inactive(fomdgt3_1),438). orientation(fomdgt3_2,42,438). coords(fomdgt3_2,262,96,438).  visible(fomdgt3_2,438). happensAt(inactive(fomdgt3_2),438).
orientation(fomdgt3_1,150,439). coords(fomdgt3_1,46,102,439).  visible(fomdgt3_1,439). happensAt(inactive(fomdgt3_1),439). orientation(fomdgt3_2,42,439). coords(fomdgt3_2,262,96,439).  visible(fomdgt3_2,439). happensAt(inactive(fomdgt3_2),439).
orientation(fomdgt3_1,150,440). coords(fomdgt3_1,46,102,440).  visible(fomdgt3_1,440). happensAt(inactive(fomdgt3_1),440). orientation(fomdgt3_2,42,440). coords(fomdgt3_2,262,96,440).  visible(fomdgt3_2,440). happensAt(inactive(fomdgt3_2),440).
orientation(fomdgt3_1,150,441). coords(fomdgt3_1,46,102,441).  visible(fomdgt3_1,441). happensAt(inactive(fomdgt3_1),441). orientation(fomdgt3_2,42,441). coords(fomdgt3_2,262,96,441).  visible(fomdgt3_2,441). happensAt(inactive(fomdgt3_2),441).
orientation(fomdgt3_1,150,442). coords(fomdgt3_1,46,102,442).  visible(fomdgt3_1,442). happensAt(inactive(fomdgt3_1),442). orientation(fomdgt3_2,42,442). coords(fomdgt3_2,262,96,442).  visible(fomdgt3_2,442). happensAt(inactive(fomdgt3_2),442).
orientation(fomdgt3_1,150,443). coords(fomdgt3_1,46,102,443).  visible(fomdgt3_1,443). happensAt(inactive(fomdgt3_1),443). orientation(fomdgt3_2,42,443). coords(fomdgt3_2,262,96,443).  visible(fomdgt3_2,443). happensAt(inactive(fomdgt3_2),443).
orientation(fomdgt3_1,150,444). coords(fomdgt3_1,46,102,444).  visible(fomdgt3_1,444). happensAt(inactive(fomdgt3_1),444). orientation(fomdgt3_2,42,444). coords(fomdgt3_2,262,96,444).  visible(fomdgt3_2,444). happensAt(inactive(fomdgt3_2),444).
#show holdsAt/2.