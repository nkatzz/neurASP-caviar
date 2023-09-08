
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
orientation(fomdgt3_6,160,447). coords(fomdgt3_6,49,103,447).  visible(fomdgt3_6,447). happensAt(walking(fomdgt3_6),447). orientation(fomdgt3_7,150,447). coords(fomdgt3_7,90,136,447).  visible(fomdgt3_7,447). happensAt(walking(fomdgt3_7),447).
orientation(fomdgt3_6,160,448). coords(fomdgt3_6,49,103,448).  visible(fomdgt3_6,448). happensAt(walking(fomdgt3_6),448). orientation(fomdgt3_7,150,448). coords(fomdgt3_7,90,136,448).  visible(fomdgt3_7,448). happensAt(walking(fomdgt3_7),448).
orientation(fomdgt3_6,160,449). coords(fomdgt3_6,49,103,449).  visible(fomdgt3_6,449). happensAt(walking(fomdgt3_6),449). orientation(fomdgt3_7,150,449). coords(fomdgt3_7,90,136,449).  visible(fomdgt3_7,449). happensAt(walking(fomdgt3_7),449).
orientation(fomdgt3_6,160,450). coords(fomdgt3_6,49,103,450).  visible(fomdgt3_6,450). happensAt(walking(fomdgt3_6),450). orientation(fomdgt3_7,150,450). coords(fomdgt3_7,91,135,450).  visible(fomdgt3_7,450). happensAt(walking(fomdgt3_7),450).
orientation(fomdgt3_6,160,451). coords(fomdgt3_6,49,103,451).  visible(fomdgt3_6,451). happensAt(walking(fomdgt3_6),451). orientation(fomdgt3_7,150,451). coords(fomdgt3_7,92,136,451).  visible(fomdgt3_7,451). happensAt(walking(fomdgt3_7),451).
orientation(fomdgt3_6,160,452). coords(fomdgt3_6,49,103,452).  visible(fomdgt3_6,452). happensAt(walking(fomdgt3_6),452). orientation(fomdgt3_7,150,452). coords(fomdgt3_7,94,136,452).  visible(fomdgt3_7,452). happensAt(walking(fomdgt3_7),452).
orientation(fomdgt3_6,160,453). coords(fomdgt3_6,49,103,453).  visible(fomdgt3_6,453). happensAt(walking(fomdgt3_6),453). orientation(fomdgt3_7,150,453). coords(fomdgt3_7,95,136,453).  visible(fomdgt3_7,453). happensAt(walking(fomdgt3_7),453).
orientation(fomdgt3_6,160,454). coords(fomdgt3_6,49,103,454).  visible(fomdgt3_6,454). happensAt(walking(fomdgt3_6),454). orientation(fomdgt3_7,150,454). coords(fomdgt3_7,97,136,454).  visible(fomdgt3_7,454). happensAt(walking(fomdgt3_7),454).
orientation(fomdgt3_6,160,455). coords(fomdgt3_6,49,103,455).  visible(fomdgt3_6,455). happensAt(walking(fomdgt3_6),455). orientation(fomdgt3_7,150,455). coords(fomdgt3_7,97,137,455).  visible(fomdgt3_7,455). happensAt(walking(fomdgt3_7),455).
orientation(fomdgt3_6,160,456). coords(fomdgt3_6,49,103,456).  visible(fomdgt3_6,456). happensAt(walking(fomdgt3_6),456). orientation(fomdgt3_7,150,456). coords(fomdgt3_7,98,137,456).  visible(fomdgt3_7,456). happensAt(walking(fomdgt3_7),456).
orientation(fomdgt3_6,160,457). coords(fomdgt3_6,49,103,457).  visible(fomdgt3_6,457). happensAt(walking(fomdgt3_6),457). orientation(fomdgt3_7,150,457). coords(fomdgt3_7,99,137,457).  visible(fomdgt3_7,457). happensAt(walking(fomdgt3_7),457).
orientation(fomdgt3_6,160,458). coords(fomdgt3_6,49,103,458).  visible(fomdgt3_6,458). happensAt(walking(fomdgt3_6),458). orientation(fomdgt3_7,150,458). coords(fomdgt3_7,100,137,458).  visible(fomdgt3_7,458). happensAt(walking(fomdgt3_7),458).
#show holdsAt/2.