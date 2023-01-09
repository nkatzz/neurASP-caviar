
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

	orientation(wk1gt_1,150,428). coords(wk1gt_1,94,98,428).  visible(wk1gt_1,428). happensAt(walking(wk1gt_1),428). orientation(wk1gt_5,162,428). coords(wk1gt_5,50,163,428).  visible(wk1gt_5,428). happensAt(active(wk1gt_5),428).
orientation(wk1gt_1,150,429). coords(wk1gt_1,93,97,429).  visible(wk1gt_1,429). happensAt(walking(wk1gt_1),429). orientation(wk1gt_5,162,429). coords(wk1gt_5,50,163,429).  visible(wk1gt_5,429). happensAt(active(wk1gt_5),429).
orientation(wk1gt_1,148,430). coords(wk1gt_1,93,97,430).  visible(wk1gt_1,430). happensAt(walking(wk1gt_1),430). orientation(wk1gt_5,162,430). coords(wk1gt_5,50,163,430).  visible(wk1gt_5,430). happensAt(active(wk1gt_5),430).
orientation(wk1gt_1,148,431). coords(wk1gt_1,93,97,431).  visible(wk1gt_1,431). happensAt(walking(wk1gt_1),431). orientation(wk1gt_5,162,431). coords(wk1gt_5,50,163,431).  visible(wk1gt_5,431). happensAt(active(wk1gt_5),431).
orientation(wk1gt_1,148,432). coords(wk1gt_1,91,97,432).  visible(wk1gt_1,432). happensAt(walking(wk1gt_1),432). orientation(wk1gt_5,162,432). coords(wk1gt_5,50,163,432).  visible(wk1gt_5,432). happensAt(active(wk1gt_5),432).
orientation(wk1gt_1,148,433). coords(wk1gt_1,91,95,433).  visible(wk1gt_1,433). happensAt(walking(wk1gt_1),433). orientation(wk1gt_5,162,433). coords(wk1gt_5,50,163,433).  visible(wk1gt_5,433). happensAt(active(wk1gt_5),433).
orientation(wk1gt_1,148,434). coords(wk1gt_1,91,95,434).  visible(wk1gt_1,434). happensAt(walking(wk1gt_1),434). orientation(wk1gt_5,162,434). coords(wk1gt_5,51,162,434).  visible(wk1gt_5,434). happensAt(active(wk1gt_5),434).
orientation(wk1gt_1,148,435). coords(wk1gt_1,91,95,435).  visible(wk1gt_1,435). happensAt(walking(wk1gt_1),435). orientation(wk1gt_5,162,435). coords(wk1gt_5,51,162,435).  visible(wk1gt_5,435). happensAt(active(wk1gt_5),435).
orientation(wk1gt_1,148,436). coords(wk1gt_1,91,94,436).  visible(wk1gt_1,436). happensAt(walking(wk1gt_1),436). orientation(wk1gt_5,162,436). coords(wk1gt_5,52,160,436).  visible(wk1gt_5,436). happensAt(active(wk1gt_5),436).
orientation(wk1gt_1,148,437). coords(wk1gt_1,91,94,437).  visible(wk1gt_1,437). happensAt(walking(wk1gt_1),437). orientation(wk1gt_5,162,437). coords(wk1gt_5,52,160,437).  visible(wk1gt_5,437). happensAt(active(wk1gt_5),437).
orientation(wk1gt_1,148,438). coords(wk1gt_1,90,93,438).  visible(wk1gt_1,438). happensAt(walking(wk1gt_1),438). orientation(wk1gt_5,162,438). coords(wk1gt_5,52,160,438).  visible(wk1gt_5,438). happensAt(active(wk1gt_5),438).
orientation(wk1gt_1,148,439). coords(wk1gt_1,89,92,439).  visible(wk1gt_1,439). happensAt(walking(wk1gt_1),439). orientation(wk1gt_5,162,439). coords(wk1gt_5,52,160,439).  visible(wk1gt_5,439). happensAt(active(wk1gt_5),439).
orientation(wk1gt_1,148,440). coords(wk1gt_1,89,92,440).  visible(wk1gt_1,440). happensAt(walking(wk1gt_1),440). orientation(wk1gt_5,162,440). coords(wk1gt_5,52,160,440).  visible(wk1gt_5,440). happensAt(active(wk1gt_5),440).
orientation(wk1gt_1,148,441). coords(wk1gt_1,89,92,441).  visible(wk1gt_1,441). happensAt(walking(wk1gt_1),441). orientation(wk1gt_5,162,441). coords(wk1gt_5,53,160,441).  visible(wk1gt_5,441). happensAt(active(wk1gt_5),441).
orientation(wk1gt_1,148,442). coords(wk1gt_1,89,92,442).  visible(wk1gt_1,442). happensAt(walking(wk1gt_1),442). orientation(wk1gt_5,162,442). coords(wk1gt_5,54,160,442).  visible(wk1gt_5,442). happensAt(active(wk1gt_5),442).
orientation(wk1gt_1,148,443). coords(wk1gt_1,89,92,443).  visible(wk1gt_1,443). happensAt(walking(wk1gt_1),443). orientation(wk1gt_5,162,443). coords(wk1gt_5,54,160,443).  visible(wk1gt_5,443). happensAt(active(wk1gt_5),443).
orientation(wk1gt_1,148,444). coords(wk1gt_1,89,92,444).  visible(wk1gt_1,444). happensAt(walking(wk1gt_1),444). orientation(wk1gt_5,162,444). coords(wk1gt_5,54,160,444).  visible(wk1gt_5,444). happensAt(walking(wk1gt_5),444).
orientation(wk1gt_1,148,445). coords(wk1gt_1,88,91,445).  visible(wk1gt_1,445). happensAt(walking(wk1gt_1),445). orientation(wk1gt_5,162,445). coords(wk1gt_5,54,160,445).  visible(wk1gt_5,445). happensAt(walking(wk1gt_5),445).
orientation(wk1gt_1,148,446). coords(wk1gt_1,88,91,446).  visible(wk1gt_1,446). happensAt(walking(wk1gt_1),446). orientation(wk1gt_5,162,446). coords(wk1gt_5,54,159,446).  visible(wk1gt_5,446). happensAt(walking(wk1gt_5),446).
orientation(wk1gt_1,148,447). coords(wk1gt_1,88,91,447).  visible(wk1gt_1,447). happensAt(walking(wk1gt_1),447). orientation(wk1gt_5,162,447). coords(wk1gt_5,54,159,447).  visible(wk1gt_5,447). happensAt(walking(wk1gt_5),447).
orientation(wk1gt_1,148,448). coords(wk1gt_1,88,91,448).  visible(wk1gt_1,448). happensAt(walking(wk1gt_1),448). orientation(wk1gt_5,162,448). coords(wk1gt_5,54,159,448).  visible(wk1gt_5,448). happensAt(walking(wk1gt_5),448).
orientation(wk1gt_1,148,449). coords(wk1gt_1,86,90,449).  visible(wk1gt_1,449). happensAt(walking(wk1gt_1),449). orientation(wk1gt_5,162,449). coords(wk1gt_5,54,159,449).  visible(wk1gt_5,449). happensAt(walking(wk1gt_5),449).
orientation(wk1gt_1,148,450). coords(wk1gt_1,86,90,450).  visible(wk1gt_1,450). happensAt(walking(wk1gt_1),450). orientation(wk1gt_5,162,450). coords(wk1gt_5,54,158,450).  visible(wk1gt_5,450). happensAt(walking(wk1gt_5),450).
orientation(wk1gt_1,148,451). coords(wk1gt_1,86,90,451).  visible(wk1gt_1,451). happensAt(walking(wk1gt_1),451). orientation(wk1gt_5,162,451). coords(wk1gt_5,54,158,451).  visible(wk1gt_5,451). happensAt(walking(wk1gt_5),451).
orientation(wk1gt_1,148,452). coords(wk1gt_1,86,90,452).  visible(wk1gt_1,452). happensAt(walking(wk1gt_1),452). orientation(wk1gt_5,162,452). coords(wk1gt_5,54,158,452).  visible(wk1gt_5,452). happensAt(walking(wk1gt_5),452).
#show holdsAt/2.