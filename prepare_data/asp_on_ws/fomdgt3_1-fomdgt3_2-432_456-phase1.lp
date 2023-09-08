
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
orientation(fomdgt3_1,150,445). coords(fomdgt3_1,46,102,445).  visible(fomdgt3_1,445). happensAt(inactive(fomdgt3_1),445). orientation(fomdgt3_2,42,445). coords(fomdgt3_2,262,96,445).  visible(fomdgt3_2,445). happensAt(inactive(fomdgt3_2),445).
orientation(fomdgt3_1,150,446). coords(fomdgt3_1,46,102,446).  visible(fomdgt3_1,446). happensAt(inactive(fomdgt3_1),446). orientation(fomdgt3_2,42,446). coords(fomdgt3_2,262,96,446).  visible(fomdgt3_2,446). happensAt(inactive(fomdgt3_2),446).
orientation(fomdgt3_1,150,447). coords(fomdgt3_1,46,102,447).  visible(fomdgt3_1,447). happensAt(inactive(fomdgt3_1),447). orientation(fomdgt3_2,42,447). coords(fomdgt3_2,262,96,447).  visible(fomdgt3_2,447). happensAt(inactive(fomdgt3_2),447).
orientation(fomdgt3_1,150,448). coords(fomdgt3_1,46,102,448).  visible(fomdgt3_1,448). happensAt(inactive(fomdgt3_1),448). orientation(fomdgt3_2,42,448). coords(fomdgt3_2,262,96,448).  visible(fomdgt3_2,448). happensAt(inactive(fomdgt3_2),448).
orientation(fomdgt3_1,150,449). coords(fomdgt3_1,46,102,449).  visible(fomdgt3_1,449). happensAt(inactive(fomdgt3_1),449). orientation(fomdgt3_2,42,449). coords(fomdgt3_2,262,96,449).  visible(fomdgt3_2,449). happensAt(inactive(fomdgt3_2),449).
orientation(fomdgt3_1,150,450). coords(fomdgt3_1,46,102,450).  visible(fomdgt3_1,450). happensAt(inactive(fomdgt3_1),450). orientation(fomdgt3_2,42,450). coords(fomdgt3_2,262,96,450).  visible(fomdgt3_2,450). happensAt(inactive(fomdgt3_2),450).
orientation(fomdgt3_1,150,451). coords(fomdgt3_1,46,102,451).  visible(fomdgt3_1,451). happensAt(inactive(fomdgt3_1),451). orientation(fomdgt3_2,42,451). coords(fomdgt3_2,262,96,451).  visible(fomdgt3_2,451). happensAt(inactive(fomdgt3_2),451).
orientation(fomdgt3_1,150,452). coords(fomdgt3_1,46,102,452).  visible(fomdgt3_1,452). happensAt(inactive(fomdgt3_1),452). orientation(fomdgt3_2,42,452). coords(fomdgt3_2,262,96,452).  visible(fomdgt3_2,452). happensAt(inactive(fomdgt3_2),452).
orientation(fomdgt3_1,150,453). coords(fomdgt3_1,46,102,453).  visible(fomdgt3_1,453). happensAt(inactive(fomdgt3_1),453). orientation(fomdgt3_2,42,453). coords(fomdgt3_2,262,96,453).  visible(fomdgt3_2,453). happensAt(inactive(fomdgt3_2),453).
orientation(fomdgt3_1,150,454). coords(fomdgt3_1,46,102,454).  visible(fomdgt3_1,454). happensAt(inactive(fomdgt3_1),454). orientation(fomdgt3_2,42,454). coords(fomdgt3_2,262,96,454).  visible(fomdgt3_2,454). happensAt(inactive(fomdgt3_2),454).
orientation(fomdgt3_1,150,455). coords(fomdgt3_1,46,102,455).  visible(fomdgt3_1,455). happensAt(inactive(fomdgt3_1),455). orientation(fomdgt3_2,42,455). coords(fomdgt3_2,262,96,455).  visible(fomdgt3_2,455). happensAt(inactive(fomdgt3_2),455).
orientation(fomdgt3_1,150,456). coords(fomdgt3_1,46,102,456).  visible(fomdgt3_1,456). happensAt(inactive(fomdgt3_1),456). orientation(fomdgt3_2,42,456). coords(fomdgt3_2,262,96,456).  visible(fomdgt3_2,456). happensAt(inactive(fomdgt3_2),456).
#show holdsAt/2.