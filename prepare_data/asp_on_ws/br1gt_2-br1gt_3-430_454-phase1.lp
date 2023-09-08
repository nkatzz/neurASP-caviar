
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

	orientation(br1gt_2,12,430). coords(br1gt_2,86,208,430).  visible(br1gt_2,430). happensAt(inactive(br1gt_2),430). orientation(br1gt_3,162,430). coords(br1gt_3,56,154,430).  visible(br1gt_3,430). happensAt(walking(br1gt_3),430).
orientation(br1gt_2,12,431). coords(br1gt_2,86,208,431).  visible(br1gt_2,431). happensAt(inactive(br1gt_2),431). orientation(br1gt_3,162,431). coords(br1gt_3,56,154,431).  visible(br1gt_3,431). happensAt(walking(br1gt_3),431).
orientation(br1gt_2,12,432). coords(br1gt_2,86,208,432).  visible(br1gt_2,432). happensAt(inactive(br1gt_2),432). orientation(br1gt_3,162,432). coords(br1gt_3,56,153,432).  visible(br1gt_3,432). happensAt(walking(br1gt_3),432).
orientation(br1gt_2,12,433). coords(br1gt_2,86,208,433).  visible(br1gt_2,433). happensAt(inactive(br1gt_2),433). orientation(br1gt_3,162,433). coords(br1gt_3,57,153,433).  visible(br1gt_3,433). happensAt(walking(br1gt_3),433).
orientation(br1gt_2,12,434). coords(br1gt_2,86,208,434).  visible(br1gt_2,434). happensAt(inactive(br1gt_2),434). orientation(br1gt_3,162,434). coords(br1gt_3,59,153,434).  visible(br1gt_3,434). happensAt(walking(br1gt_3),434).
orientation(br1gt_2,12,435). coords(br1gt_2,86,208,435).  visible(br1gt_2,435). happensAt(inactive(br1gt_2),435). orientation(br1gt_3,162,435). coords(br1gt_3,59,153,435).  visible(br1gt_3,435). happensAt(walking(br1gt_3),435).
orientation(br1gt_2,12,436). coords(br1gt_2,86,208,436).  visible(br1gt_2,436). happensAt(inactive(br1gt_2),436). orientation(br1gt_3,162,436). coords(br1gt_3,59,153,436).  visible(br1gt_3,436). happensAt(walking(br1gt_3),436).
orientation(br1gt_2,12,437). coords(br1gt_2,86,208,437).  visible(br1gt_2,437). happensAt(inactive(br1gt_2),437). orientation(br1gt_3,162,437). coords(br1gt_3,60,153,437).  visible(br1gt_3,437). happensAt(walking(br1gt_3),437).
orientation(br1gt_2,12,438). coords(br1gt_2,86,208,438).  visible(br1gt_2,438). happensAt(inactive(br1gt_2),438). orientation(br1gt_3,162,438). coords(br1gt_3,61,153,438).  visible(br1gt_3,438). happensAt(walking(br1gt_3),438).
orientation(br1gt_2,12,439). coords(br1gt_2,86,208,439).  visible(br1gt_2,439). happensAt(inactive(br1gt_2),439). orientation(br1gt_3,162,439). coords(br1gt_3,62,153,439).  visible(br1gt_3,439). happensAt(walking(br1gt_3),439).
orientation(br1gt_2,12,440). coords(br1gt_2,86,208,440).  visible(br1gt_2,440). happensAt(inactive(br1gt_2),440). orientation(br1gt_3,162,440). coords(br1gt_3,63,152,440).  visible(br1gt_3,440). happensAt(walking(br1gt_3),440).
orientation(br1gt_2,12,441). coords(br1gt_2,86,208,441).  visible(br1gt_2,441). happensAt(inactive(br1gt_2),441). orientation(br1gt_3,162,441). coords(br1gt_3,63,152,441).  visible(br1gt_3,441). happensAt(walking(br1gt_3),441).
orientation(br1gt_2,12,442). coords(br1gt_2,86,208,442).  visible(br1gt_2,442). happensAt(inactive(br1gt_2),442). orientation(br1gt_3,162,442). coords(br1gt_3,63,151,442).  visible(br1gt_3,442). happensAt(walking(br1gt_3),442).
orientation(br1gt_2,12,443). coords(br1gt_2,86,208,443).  visible(br1gt_2,443). happensAt(inactive(br1gt_2),443). orientation(br1gt_3,162,443). coords(br1gt_3,63,151,443).  visible(br1gt_3,443). happensAt(walking(br1gt_3),443).
orientation(br1gt_2,12,444). coords(br1gt_2,86,208,444).  visible(br1gt_2,444). happensAt(inactive(br1gt_2),444). orientation(br1gt_3,162,444). coords(br1gt_3,63,151,444).  visible(br1gt_3,444). happensAt(walking(br1gt_3),444).
orientation(br1gt_2,12,445). coords(br1gt_2,86,208,445).  visible(br1gt_2,445). happensAt(inactive(br1gt_2),445). orientation(br1gt_3,160,445). coords(br1gt_3,63,150,445).  visible(br1gt_3,445). happensAt(walking(br1gt_3),445).
orientation(br1gt_2,12,446). coords(br1gt_2,86,208,446).  visible(br1gt_2,446). happensAt(inactive(br1gt_2),446). orientation(br1gt_3,160,446). coords(br1gt_3,64,150,446).  visible(br1gt_3,446). happensAt(walking(br1gt_3),446).
orientation(br1gt_2,12,447). coords(br1gt_2,86,208,447).  visible(br1gt_2,447). happensAt(inactive(br1gt_2),447). orientation(br1gt_3,160,447). coords(br1gt_3,65,149,447).  visible(br1gt_3,447). happensAt(walking(br1gt_3),447).
orientation(br1gt_2,12,448). coords(br1gt_2,86,208,448).  visible(br1gt_2,448). happensAt(inactive(br1gt_2),448). orientation(br1gt_3,160,448). coords(br1gt_3,66,149,448).  visible(br1gt_3,448). happensAt(walking(br1gt_3),448).
orientation(br1gt_2,12,449). coords(br1gt_2,86,208,449).  visible(br1gt_2,449). happensAt(inactive(br1gt_2),449). orientation(br1gt_3,160,449). coords(br1gt_3,67,147,449).  visible(br1gt_3,449). happensAt(walking(br1gt_3),449).
orientation(br1gt_2,12,450). coords(br1gt_2,86,208,450).  visible(br1gt_2,450). happensAt(inactive(br1gt_2),450). orientation(br1gt_3,160,450). coords(br1gt_3,69,147,450).  visible(br1gt_3,450). happensAt(walking(br1gt_3),450).
orientation(br1gt_2,12,451). coords(br1gt_2,86,208,451).  visible(br1gt_2,451). happensAt(inactive(br1gt_2),451). orientation(br1gt_3,160,451). coords(br1gt_3,69,147,451).  visible(br1gt_3,451). happensAt(walking(br1gt_3),451).
orientation(br1gt_2,12,452). coords(br1gt_2,86,208,452).  visible(br1gt_2,452). happensAt(inactive(br1gt_2),452). orientation(br1gt_3,160,452). coords(br1gt_3,69,147,452).  visible(br1gt_3,452). happensAt(walking(br1gt_3),452).
orientation(br1gt_2,12,453). coords(br1gt_2,86,208,453).  visible(br1gt_2,453). happensAt(inactive(br1gt_2),453). orientation(br1gt_3,160,453). coords(br1gt_3,70,146,453).  visible(br1gt_3,453). happensAt(walking(br1gt_3),453).
orientation(br1gt_2,12,454). coords(br1gt_2,86,208,454).  visible(br1gt_2,454). happensAt(inactive(br1gt_2),454). orientation(br1gt_3,160,454). coords(br1gt_3,70,146,454).  visible(br1gt_3,454). happensAt(walking(br1gt_3),454).
#show holdsAt/2.