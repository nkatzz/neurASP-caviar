
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

	orientation(wk2gt_1,125,427). coords(wk2gt_1,115,53,427).  visible(wk2gt_1,427). happensAt(inactive(wk2gt_1),427). orientation(wk2gt_4,149,427). coords(wk2gt_4,107,103,427).  visible(wk2gt_4,427). happensAt(walking(wk2gt_4),427).
orientation(wk2gt_1,125,428). coords(wk2gt_1,115,53,428).  visible(wk2gt_1,428). happensAt(inactive(wk2gt_1),428). orientation(wk2gt_4,149,428). coords(wk2gt_4,105,103,428).  visible(wk2gt_4,428). happensAt(walking(wk2gt_4),428).
orientation(wk2gt_1,125,429). coords(wk2gt_1,115,53,429).  visible(wk2gt_1,429). happensAt(inactive(wk2gt_1),429). orientation(wk2gt_4,149,429). coords(wk2gt_4,105,102,429).  visible(wk2gt_4,429). happensAt(walking(wk2gt_4),429).
orientation(wk2gt_1,125,430). coords(wk2gt_1,115,53,430).  visible(wk2gt_1,430). happensAt(inactive(wk2gt_1),430). orientation(wk2gt_4,149,430). coords(wk2gt_4,104,102,430).  visible(wk2gt_4,430). happensAt(walking(wk2gt_4),430).
orientation(wk2gt_1,125,431). coords(wk2gt_1,115,53,431).  visible(wk2gt_1,431). happensAt(inactive(wk2gt_1),431). orientation(wk2gt_4,149,431). coords(wk2gt_4,104,101,431).  visible(wk2gt_4,431). happensAt(walking(wk2gt_4),431).
orientation(wk2gt_1,125,432). coords(wk2gt_1,115,53,432).  visible(wk2gt_1,432). happensAt(inactive(wk2gt_1),432). orientation(wk2gt_4,149,432). coords(wk2gt_4,104,101,432).  visible(wk2gt_4,432). happensAt(walking(wk2gt_4),432).
orientation(wk2gt_1,125,433). coords(wk2gt_1,115,53,433).  visible(wk2gt_1,433). happensAt(inactive(wk2gt_1),433). orientation(wk2gt_4,149,433). coords(wk2gt_4,104,101,433).  visible(wk2gt_4,433). happensAt(walking(wk2gt_4),433).
orientation(wk2gt_1,125,434). coords(wk2gt_1,115,53,434).  visible(wk2gt_1,434). happensAt(inactive(wk2gt_1),434). orientation(wk2gt_4,149,434). coords(wk2gt_4,103,100,434).  visible(wk2gt_4,434). happensAt(walking(wk2gt_4),434).
orientation(wk2gt_1,125,435). coords(wk2gt_1,115,53,435).  visible(wk2gt_1,435). happensAt(inactive(wk2gt_1),435). orientation(wk2gt_4,149,435). coords(wk2gt_4,103,100,435).  visible(wk2gt_4,435). happensAt(walking(wk2gt_4),435).
orientation(wk2gt_1,125,436). coords(wk2gt_1,115,53,436).  visible(wk2gt_1,436). happensAt(inactive(wk2gt_1),436). orientation(wk2gt_4,149,436). coords(wk2gt_4,102,100,436).  visible(wk2gt_4,436). happensAt(walking(wk2gt_4),436).
orientation(wk2gt_1,125,437). coords(wk2gt_1,115,53,437).  visible(wk2gt_1,437). happensAt(inactive(wk2gt_1),437). orientation(wk2gt_4,149,437). coords(wk2gt_4,101,100,437).  visible(wk2gt_4,437). happensAt(walking(wk2gt_4),437).
orientation(wk2gt_1,125,438). coords(wk2gt_1,115,53,438).  visible(wk2gt_1,438). happensAt(inactive(wk2gt_1),438). orientation(wk2gt_4,149,438). coords(wk2gt_4,101,100,438).  visible(wk2gt_4,438). happensAt(walking(wk2gt_4),438).
orientation(wk2gt_1,125,439). coords(wk2gt_1,115,53,439).  visible(wk2gt_1,439). happensAt(inactive(wk2gt_1),439). orientation(wk2gt_4,149,439). coords(wk2gt_4,99,100,439).  visible(wk2gt_4,439). happensAt(walking(wk2gt_4),439).
orientation(wk2gt_1,125,440). coords(wk2gt_1,115,53,440).  visible(wk2gt_1,440). happensAt(inactive(wk2gt_1),440). orientation(wk2gt_4,149,440). coords(wk2gt_4,97,100,440).  visible(wk2gt_4,440). happensAt(walking(wk2gt_4),440).
orientation(wk2gt_1,125,441). coords(wk2gt_1,115,53,441).  visible(wk2gt_1,441). happensAt(inactive(wk2gt_1),441). orientation(wk2gt_4,149,441). coords(wk2gt_4,97,100,441).  visible(wk2gt_4,441). happensAt(walking(wk2gt_4),441).
orientation(wk2gt_1,125,442). coords(wk2gt_1,115,53,442).  visible(wk2gt_1,442). happensAt(inactive(wk2gt_1),442). orientation(wk2gt_4,149,442). coords(wk2gt_4,95,100,442).  visible(wk2gt_4,442). happensAt(walking(wk2gt_4),442).
orientation(wk2gt_1,125,443). coords(wk2gt_1,115,53,443).  visible(wk2gt_1,443). happensAt(inactive(wk2gt_1),443). orientation(wk2gt_4,149,443). coords(wk2gt_4,95,100,443).  visible(wk2gt_4,443). happensAt(walking(wk2gt_4),443).
orientation(wk2gt_1,125,444). coords(wk2gt_1,115,53,444).  visible(wk2gt_1,444). happensAt(inactive(wk2gt_1),444). orientation(wk2gt_4,149,444). coords(wk2gt_4,94,100,444).  visible(wk2gt_4,444). happensAt(walking(wk2gt_4),444).
orientation(wk2gt_1,125,445). coords(wk2gt_1,115,53,445).  visible(wk2gt_1,445). happensAt(inactive(wk2gt_1),445). orientation(wk2gt_4,149,445). coords(wk2gt_4,94,100,445).  visible(wk2gt_4,445). happensAt(walking(wk2gt_4),445).
orientation(wk2gt_1,125,446). coords(wk2gt_1,115,53,446).  visible(wk2gt_1,446). happensAt(inactive(wk2gt_1),446). orientation(wk2gt_4,149,446). coords(wk2gt_4,94,100,446).  visible(wk2gt_4,446). happensAt(walking(wk2gt_4),446).
orientation(wk2gt_1,125,447). coords(wk2gt_1,115,53,447).  visible(wk2gt_1,447). happensAt(inactive(wk2gt_1),447). orientation(wk2gt_4,149,447). coords(wk2gt_4,94,100,447).  visible(wk2gt_4,447). happensAt(walking(wk2gt_4),447).
orientation(wk2gt_1,125,448). coords(wk2gt_1,115,53,448).  visible(wk2gt_1,448). happensAt(inactive(wk2gt_1),448). orientation(wk2gt_4,149,448). coords(wk2gt_4,94,100,448).  visible(wk2gt_4,448). happensAt(walking(wk2gt_4),448).
orientation(wk2gt_1,125,449). coords(wk2gt_1,115,53,449).  visible(wk2gt_1,449). happensAt(inactive(wk2gt_1),449). orientation(wk2gt_4,149,449). coords(wk2gt_4,93,100,449).  visible(wk2gt_4,449). happensAt(walking(wk2gt_4),449).
orientation(wk2gt_1,125,450). coords(wk2gt_1,115,53,450).  visible(wk2gt_1,450). happensAt(inactive(wk2gt_1),450). orientation(wk2gt_4,149,450). coords(wk2gt_4,91,100,450).  visible(wk2gt_4,450). happensAt(walking(wk2gt_4),450).
orientation(wk2gt_1,125,451). coords(wk2gt_1,115,53,451).  visible(wk2gt_1,451). happensAt(inactive(wk2gt_1),451). orientation(wk2gt_4,149,451). coords(wk2gt_4,90,99,451).  visible(wk2gt_4,451). happensAt(walking(wk2gt_4),451).
#show holdsAt/2.