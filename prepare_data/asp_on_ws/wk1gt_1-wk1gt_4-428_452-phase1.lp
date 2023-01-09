
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

	orientation(wk1gt_1,150,428). coords(wk1gt_1,94,98,428).  visible(wk1gt_1,428). happensAt(walking(wk1gt_1),428). orientation(wk1gt_4,147,428). coords(wk1gt_4,54,172,428).  visible(wk1gt_4,428). happensAt(active(wk1gt_4),428).
orientation(wk1gt_1,150,429). coords(wk1gt_1,93,97,429).  visible(wk1gt_1,429). happensAt(walking(wk1gt_1),429). orientation(wk1gt_4,147,429). coords(wk1gt_4,54,172,429).  visible(wk1gt_4,429). happensAt(active(wk1gt_4),429).
orientation(wk1gt_1,148,430). coords(wk1gt_1,93,97,430).  visible(wk1gt_1,430). happensAt(walking(wk1gt_1),430). orientation(wk1gt_4,147,430). coords(wk1gt_4,54,171,430).  visible(wk1gt_4,430). happensAt(active(wk1gt_4),430).
orientation(wk1gt_1,148,431). coords(wk1gt_1,93,97,431).  visible(wk1gt_1,431). happensAt(walking(wk1gt_1),431). orientation(wk1gt_4,147,431). coords(wk1gt_4,54,171,431).  visible(wk1gt_4,431). happensAt(active(wk1gt_4),431).
orientation(wk1gt_1,148,432). coords(wk1gt_1,91,97,432).  visible(wk1gt_1,432). happensAt(walking(wk1gt_1),432). orientation(wk1gt_4,147,432). coords(wk1gt_4,54,171,432).  visible(wk1gt_4,432). happensAt(active(wk1gt_4),432).
orientation(wk1gt_1,148,433). coords(wk1gt_1,91,95,433).  visible(wk1gt_1,433). happensAt(walking(wk1gt_1),433). orientation(wk1gt_4,147,433). coords(wk1gt_4,54,171,433).  visible(wk1gt_4,433). happensAt(active(wk1gt_4),433).
orientation(wk1gt_1,148,434). coords(wk1gt_1,91,95,434).  visible(wk1gt_1,434). happensAt(walking(wk1gt_1),434). orientation(wk1gt_4,147,434). coords(wk1gt_4,54,170,434).  visible(wk1gt_4,434). happensAt(active(wk1gt_4),434).
orientation(wk1gt_1,148,435). coords(wk1gt_1,91,95,435).  visible(wk1gt_1,435). happensAt(walking(wk1gt_1),435). orientation(wk1gt_4,147,435). coords(wk1gt_4,55,170,435).  visible(wk1gt_4,435). happensAt(active(wk1gt_4),435).
orientation(wk1gt_1,148,436). coords(wk1gt_1,91,94,436).  visible(wk1gt_1,436). happensAt(walking(wk1gt_1),436). orientation(wk1gt_4,147,436). coords(wk1gt_4,55,169,436).  visible(wk1gt_4,436). happensAt(active(wk1gt_4),436).
orientation(wk1gt_1,148,437). coords(wk1gt_1,91,94,437).  visible(wk1gt_1,437). happensAt(walking(wk1gt_1),437). orientation(wk1gt_4,147,437). coords(wk1gt_4,56,169,437).  visible(wk1gt_4,437). happensAt(active(wk1gt_4),437).
orientation(wk1gt_1,148,438). coords(wk1gt_1,90,93,438).  visible(wk1gt_1,438). happensAt(walking(wk1gt_1),438). orientation(wk1gt_4,147,438). coords(wk1gt_4,56,169,438).  visible(wk1gt_4,438). happensAt(active(wk1gt_4),438).
orientation(wk1gt_1,148,439). coords(wk1gt_1,89,92,439).  visible(wk1gt_1,439). happensAt(walking(wk1gt_1),439). orientation(wk1gt_4,147,439). coords(wk1gt_4,56,169,439).  visible(wk1gt_4,439). happensAt(active(wk1gt_4),439).
orientation(wk1gt_1,148,440). coords(wk1gt_1,89,92,440).  visible(wk1gt_1,440). happensAt(walking(wk1gt_1),440). orientation(wk1gt_4,147,440). coords(wk1gt_4,56,169,440).  visible(wk1gt_4,440). happensAt(active(wk1gt_4),440).
orientation(wk1gt_1,148,441). coords(wk1gt_1,89,92,441).  visible(wk1gt_1,441). happensAt(walking(wk1gt_1),441). orientation(wk1gt_4,147,441). coords(wk1gt_4,56,169,441).  visible(wk1gt_4,441). happensAt(active(wk1gt_4),441).
orientation(wk1gt_1,148,442). coords(wk1gt_1,89,92,442).  visible(wk1gt_1,442). happensAt(walking(wk1gt_1),442). orientation(wk1gt_4,147,442). coords(wk1gt_4,56,169,442).  visible(wk1gt_4,442). happensAt(active(wk1gt_4),442).
orientation(wk1gt_1,148,443). coords(wk1gt_1,89,92,443).  visible(wk1gt_1,443). happensAt(walking(wk1gt_1),443). orientation(wk1gt_4,147,443). coords(wk1gt_4,56,169,443).  visible(wk1gt_4,443). happensAt(active(wk1gt_4),443).
orientation(wk1gt_1,148,444). coords(wk1gt_1,89,92,444).  visible(wk1gt_1,444). happensAt(walking(wk1gt_1),444). orientation(wk1gt_4,147,444). coords(wk1gt_4,56,169,444).  visible(wk1gt_4,444). happensAt(active(wk1gt_4),444).
orientation(wk1gt_1,148,445). coords(wk1gt_1,88,91,445).  visible(wk1gt_1,445). happensAt(walking(wk1gt_1),445). orientation(wk1gt_4,147,445). coords(wk1gt_4,56,169,445).  visible(wk1gt_4,445). happensAt(active(wk1gt_4),445).
orientation(wk1gt_1,148,446). coords(wk1gt_1,88,91,446).  visible(wk1gt_1,446). happensAt(walking(wk1gt_1),446). orientation(wk1gt_4,147,446). coords(wk1gt_4,56,169,446).  visible(wk1gt_4,446). happensAt(active(wk1gt_4),446).
orientation(wk1gt_1,148,447). coords(wk1gt_1,88,91,447).  visible(wk1gt_1,447). happensAt(walking(wk1gt_1),447). orientation(wk1gt_4,147,447). coords(wk1gt_4,56,169,447).  visible(wk1gt_4,447). happensAt(active(wk1gt_4),447).
orientation(wk1gt_1,148,448). coords(wk1gt_1,88,91,448).  visible(wk1gt_1,448). happensAt(walking(wk1gt_1),448). orientation(wk1gt_4,147,448). coords(wk1gt_4,56,169,448).  visible(wk1gt_4,448). happensAt(active(wk1gt_4),448).
orientation(wk1gt_1,148,449). coords(wk1gt_1,86,90,449).  visible(wk1gt_1,449). happensAt(walking(wk1gt_1),449). orientation(wk1gt_4,147,449). coords(wk1gt_4,56,169,449).  visible(wk1gt_4,449). happensAt(active(wk1gt_4),449).
orientation(wk1gt_1,148,450). coords(wk1gt_1,86,90,450).  visible(wk1gt_1,450). happensAt(walking(wk1gt_1),450). orientation(wk1gt_4,147,450). coords(wk1gt_4,56,169,450).  visible(wk1gt_4,450). happensAt(active(wk1gt_4),450).
orientation(wk1gt_1,148,451). coords(wk1gt_1,86,90,451).  visible(wk1gt_1,451). happensAt(walking(wk1gt_1),451). orientation(wk1gt_4,147,451). coords(wk1gt_4,56,169,451).  visible(wk1gt_4,451). happensAt(active(wk1gt_4),451).
orientation(wk1gt_1,148,452). coords(wk1gt_1,86,90,452).  visible(wk1gt_1,452). happensAt(walking(wk1gt_1),452). orientation(wk1gt_4,147,452). coords(wk1gt_4,56,169,452).  visible(wk1gt_4,452). happensAt(active(wk1gt_4),452).
#show holdsAt/2.