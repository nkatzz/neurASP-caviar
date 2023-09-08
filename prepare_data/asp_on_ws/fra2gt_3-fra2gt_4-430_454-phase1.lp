
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

	orientation(fra2gt_3,164,430). coords(fra2gt_3,52,161,430).  visible(fra2gt_3,430). happensAt(walking(fra2gt_3),430). orientation(fra2gt_4,146,430). coords(fra2gt_4,73,104,430).  visible(fra2gt_4,430). happensAt(running(fra2gt_4),430).
orientation(fra2gt_3,164,431). coords(fra2gt_3,51,162,431).  visible(fra2gt_3,431). happensAt(walking(fra2gt_3),431). orientation(fra2gt_4,146,431). coords(fra2gt_4,72,103,431).  visible(fra2gt_4,431). happensAt(running(fra2gt_4),431).
orientation(fra2gt_3,164,432). coords(fra2gt_3,51,162,432).  visible(fra2gt_3,432). happensAt(walking(fra2gt_3),432). orientation(fra2gt_4,146,432). coords(fra2gt_4,72,102,432).  visible(fra2gt_4,432). happensAt(running(fra2gt_4),432).
orientation(fra2gt_3,164,433). coords(fra2gt_3,50,162,433).  visible(fra2gt_3,433). happensAt(walking(fra2gt_3),433). orientation(fra2gt_4,143,433). coords(fra2gt_4,71,101,433).  visible(fra2gt_4,433). happensAt(running(fra2gt_4),433).
orientation(fra2gt_3,164,434). coords(fra2gt_3,50,163,434).  visible(fra2gt_3,434). happensAt(walking(fra2gt_3),434). orientation(fra2gt_4,141,434). coords(fra2gt_4,71,100,434).  visible(fra2gt_4,434). happensAt(running(fra2gt_4),434).
orientation(fra2gt_3,164,435). coords(fra2gt_3,49,163,435).  visible(fra2gt_3,435). happensAt(walking(fra2gt_3),435). orientation(fra2gt_4,141,435). coords(fra2gt_4,71,99,435).  visible(fra2gt_4,435). happensAt(running(fra2gt_4),435).
orientation(fra2gt_3,164,436). coords(fra2gt_3,49,163,436).  visible(fra2gt_3,436). happensAt(walking(fra2gt_3),436). orientation(fra2gt_4,137,436). coords(fra2gt_4,72,98,436).  visible(fra2gt_4,436). happensAt(running(fra2gt_4),436).
orientation(fra2gt_3,164,437). coords(fra2gt_3,49,163,437).  visible(fra2gt_3,437). happensAt(walking(fra2gt_3),437). orientation(fra2gt_4,137,437). coords(fra2gt_4,72,96,437).  visible(fra2gt_4,437). happensAt(running(fra2gt_4),437).
orientation(fra2gt_3,164,438). coords(fra2gt_3,47,164,438).  visible(fra2gt_3,438). happensAt(walking(fra2gt_3),438). orientation(fra2gt_4,137,438). coords(fra2gt_4,72,95,438).  visible(fra2gt_4,438). happensAt(running(fra2gt_4),438).
orientation(fra2gt_3,164,439). coords(fra2gt_3,47,163,439).  visible(fra2gt_3,439). happensAt(walking(fra2gt_3),439). orientation(fra2gt_4,137,439). coords(fra2gt_4,72,93,439).  visible(fra2gt_4,439). happensAt(running(fra2gt_4),439).
orientation(fra2gt_3,164,440). coords(fra2gt_3,46,163,440).  visible(fra2gt_3,440). happensAt(walking(fra2gt_3),440). orientation(fra2gt_4,137,440). coords(fra2gt_4,72,92,440).  visible(fra2gt_4,440). happensAt(running(fra2gt_4),440).
orientation(fra2gt_3,164,441). coords(fra2gt_3,45,164,441).  visible(fra2gt_3,441). happensAt(walking(fra2gt_3),441). orientation(fra2gt_4,137,441). coords(fra2gt_4,72,91,441).  visible(fra2gt_4,441). happensAt(running(fra2gt_4),441).
orientation(fra2gt_3,164,442). coords(fra2gt_3,45,164,442).  visible(fra2gt_3,442). happensAt(walking(fra2gt_3),442). orientation(fra2gt_4,143,442). coords(fra2gt_4,72,89,442).  visible(fra2gt_4,442). happensAt(running(fra2gt_4),442).
orientation(fra2gt_3,164,443). coords(fra2gt_3,45,164,443).  visible(fra2gt_3,443). happensAt(walking(fra2gt_3),443). orientation(fra2gt_4,143,443). coords(fra2gt_4,72,88,443).  visible(fra2gt_4,443). happensAt(running(fra2gt_4),443).
orientation(fra2gt_3,164,444). coords(fra2gt_3,44,164,444).  visible(fra2gt_3,444). happensAt(walking(fra2gt_3),444). orientation(fra2gt_4,143,444). coords(fra2gt_4,72,87,444).  visible(fra2gt_4,444). happensAt(running(fra2gt_4),444).
orientation(fra2gt_3,164,445). coords(fra2gt_3,44,165,445).  visible(fra2gt_3,445). happensAt(walking(fra2gt_3),445). orientation(fra2gt_4,146,445). coords(fra2gt_4,74,85,445).  visible(fra2gt_4,445). happensAt(running(fra2gt_4),445).
orientation(fra2gt_3,164,446). coords(fra2gt_3,44,165,446).  visible(fra2gt_3,446). happensAt(walking(fra2gt_3),446). orientation(fra2gt_4,150,446). coords(fra2gt_4,74,84,446).  visible(fra2gt_4,446). happensAt(running(fra2gt_4),446).
orientation(fra2gt_3,164,447). coords(fra2gt_3,44,165,447).  visible(fra2gt_3,447). happensAt(walking(fra2gt_3),447). orientation(fra2gt_4,150,447). coords(fra2gt_4,74,84,447).  visible(fra2gt_4,447). happensAt(running(fra2gt_4),447).
orientation(fra2gt_3,164,448). coords(fra2gt_3,44,164,448).  visible(fra2gt_3,448). happensAt(walking(fra2gt_3),448). orientation(fra2gt_4,150,448). coords(fra2gt_4,74,84,448).  visible(fra2gt_4,448). happensAt(running(fra2gt_4),448).
orientation(fra2gt_3,164,449). coords(fra2gt_3,44,164,449).  visible(fra2gt_3,449). happensAt(walking(fra2gt_3),449). orientation(fra2gt_4,150,449). coords(fra2gt_4,73,83,449).  visible(fra2gt_4,449). happensAt(running(fra2gt_4),449).
orientation(fra2gt_3,164,450). coords(fra2gt_3,43,164,450).  visible(fra2gt_3,450). happensAt(walking(fra2gt_3),450). orientation(fra2gt_4,150,450). coords(fra2gt_4,73,83,450).  visible(fra2gt_4,450). happensAt(running(fra2gt_4),450).
orientation(fra2gt_3,164,451). coords(fra2gt_3,42,164,451).  visible(fra2gt_3,451). happensAt(walking(fra2gt_3),451). orientation(fra2gt_4,150,451). coords(fra2gt_4,73,82,451).  visible(fra2gt_4,451). happensAt(running(fra2gt_4),451).
orientation(fra2gt_3,164,452). coords(fra2gt_3,43,164,452).  visible(fra2gt_3,452). happensAt(walking(fra2gt_3),452). orientation(fra2gt_4,150,452). coords(fra2gt_4,72,82,452).  visible(fra2gt_4,452). happensAt(running(fra2gt_4),452).
orientation(fra2gt_3,153,453). coords(fra2gt_3,42,164,453).  visible(fra2gt_3,453). happensAt(walking(fra2gt_3),453). orientation(fra2gt_4,150,453). coords(fra2gt_4,73,81,453).  visible(fra2gt_4,453). happensAt(running(fra2gt_4),453).
orientation(fra2gt_3,153,454). coords(fra2gt_3,42,163,454).  visible(fra2gt_3,454). happensAt(walking(fra2gt_3),454). orientation(fra2gt_4,150,454). coords(fra2gt_4,72,81,454).  visible(fra2gt_4,454). happensAt(running(fra2gt_4),454).
#show holdsAt/2.