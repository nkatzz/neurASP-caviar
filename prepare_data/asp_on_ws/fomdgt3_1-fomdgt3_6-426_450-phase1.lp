
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

	orientation(fomdgt3_1,150,426). coords(fomdgt3_1,46,102,426).  visible(fomdgt3_1,426). happensAt(inactive(fomdgt3_1),426). orientation(fomdgt3_6,160,426). coords(fomdgt3_6,53,91,426).  visible(fomdgt3_6,426). happensAt(walking(fomdgt3_6),426).
orientation(fomdgt3_1,150,427). coords(fomdgt3_1,46,102,427).  visible(fomdgt3_1,427). happensAt(inactive(fomdgt3_1),427). orientation(fomdgt3_6,160,427). coords(fomdgt3_6,51,91,427).  visible(fomdgt3_6,427). happensAt(walking(fomdgt3_6),427).
orientation(fomdgt3_1,150,428). coords(fomdgt3_1,46,102,428).  visible(fomdgt3_1,428). happensAt(inactive(fomdgt3_1),428). orientation(fomdgt3_6,160,428). coords(fomdgt3_6,51,92,428).  visible(fomdgt3_6,428). happensAt(walking(fomdgt3_6),428).
orientation(fomdgt3_1,150,429). coords(fomdgt3_1,46,102,429).  visible(fomdgt3_1,429). happensAt(inactive(fomdgt3_1),429). orientation(fomdgt3_6,160,429). coords(fomdgt3_6,51,93,429).  visible(fomdgt3_6,429). happensAt(walking(fomdgt3_6),429).
orientation(fomdgt3_1,150,430). coords(fomdgt3_1,46,102,430).  visible(fomdgt3_1,430). happensAt(inactive(fomdgt3_1),430). orientation(fomdgt3_6,160,430). coords(fomdgt3_6,51,93,430).  visible(fomdgt3_6,430). happensAt(walking(fomdgt3_6),430).
orientation(fomdgt3_1,150,431). coords(fomdgt3_1,46,102,431).  visible(fomdgt3_1,431). happensAt(inactive(fomdgt3_1),431). orientation(fomdgt3_6,160,431). coords(fomdgt3_6,50,93,431).  visible(fomdgt3_6,431). happensAt(walking(fomdgt3_6),431).
orientation(fomdgt3_1,150,432). coords(fomdgt3_1,46,102,432).  visible(fomdgt3_1,432). happensAt(inactive(fomdgt3_1),432). orientation(fomdgt3_6,160,432). coords(fomdgt3_6,49,93,432).  visible(fomdgt3_6,432). happensAt(walking(fomdgt3_6),432).
orientation(fomdgt3_1,150,433). coords(fomdgt3_1,46,102,433).  visible(fomdgt3_1,433). happensAt(inactive(fomdgt3_1),433). orientation(fomdgt3_6,160,433). coords(fomdgt3_6,49,94,433).  visible(fomdgt3_6,433). happensAt(walking(fomdgt3_6),433).
orientation(fomdgt3_1,150,434). coords(fomdgt3_1,46,102,434).  visible(fomdgt3_1,434). happensAt(inactive(fomdgt3_1),434). orientation(fomdgt3_6,160,434). coords(fomdgt3_6,49,94,434).  visible(fomdgt3_6,434). happensAt(walking(fomdgt3_6),434).
orientation(fomdgt3_1,150,435). coords(fomdgt3_1,46,102,435).  visible(fomdgt3_1,435). happensAt(inactive(fomdgt3_1),435). orientation(fomdgt3_6,160,435). coords(fomdgt3_6,49,94,435).  visible(fomdgt3_6,435). happensAt(walking(fomdgt3_6),435).
orientation(fomdgt3_1,150,436). coords(fomdgt3_1,46,102,436).  visible(fomdgt3_1,436). happensAt(inactive(fomdgt3_1),436). orientation(fomdgt3_6,160,436). coords(fomdgt3_6,49,95,436).  visible(fomdgt3_6,436). happensAt(walking(fomdgt3_6),436).
orientation(fomdgt3_1,150,437). coords(fomdgt3_1,46,102,437).  visible(fomdgt3_1,437). happensAt(inactive(fomdgt3_1),437). orientation(fomdgt3_6,160,437). coords(fomdgt3_6,49,97,437).  visible(fomdgt3_6,437). happensAt(walking(fomdgt3_6),437).
orientation(fomdgt3_1,150,438). coords(fomdgt3_1,46,102,438).  visible(fomdgt3_1,438). happensAt(inactive(fomdgt3_1),438). orientation(fomdgt3_6,160,438). coords(fomdgt3_6,49,97,438).  visible(fomdgt3_6,438). happensAt(walking(fomdgt3_6),438).
orientation(fomdgt3_1,150,439). coords(fomdgt3_1,46,102,439).  visible(fomdgt3_1,439). happensAt(inactive(fomdgt3_1),439). orientation(fomdgt3_6,160,439). coords(fomdgt3_6,49,103,439).  visible(fomdgt3_6,439). happensAt(walking(fomdgt3_6),439).
orientation(fomdgt3_1,150,440). coords(fomdgt3_1,46,102,440).  visible(fomdgt3_1,440). happensAt(inactive(fomdgt3_1),440). orientation(fomdgt3_6,160,440). coords(fomdgt3_6,49,103,440).  visible(fomdgt3_6,440). happensAt(walking(fomdgt3_6),440).
orientation(fomdgt3_1,150,441). coords(fomdgt3_1,46,102,441).  visible(fomdgt3_1,441). happensAt(inactive(fomdgt3_1),441). orientation(fomdgt3_6,160,441). coords(fomdgt3_6,49,103,441).  visible(fomdgt3_6,441). happensAt(walking(fomdgt3_6),441).
orientation(fomdgt3_1,150,442). coords(fomdgt3_1,46,102,442).  visible(fomdgt3_1,442). happensAt(inactive(fomdgt3_1),442). orientation(fomdgt3_6,160,442). coords(fomdgt3_6,49,103,442).  visible(fomdgt3_6,442). happensAt(walking(fomdgt3_6),442).
orientation(fomdgt3_1,150,443). coords(fomdgt3_1,46,102,443).  visible(fomdgt3_1,443). happensAt(inactive(fomdgt3_1),443). orientation(fomdgt3_6,160,443). coords(fomdgt3_6,49,103,443).  visible(fomdgt3_6,443). happensAt(walking(fomdgt3_6),443).
orientation(fomdgt3_1,150,444). coords(fomdgt3_1,46,102,444).  visible(fomdgt3_1,444). happensAt(inactive(fomdgt3_1),444). orientation(fomdgt3_6,160,444). coords(fomdgt3_6,49,103,444).  visible(fomdgt3_6,444). happensAt(walking(fomdgt3_6),444).
orientation(fomdgt3_1,150,445). coords(fomdgt3_1,46,102,445).  visible(fomdgt3_1,445). happensAt(inactive(fomdgt3_1),445). orientation(fomdgt3_6,160,445). coords(fomdgt3_6,49,103,445).  visible(fomdgt3_6,445). happensAt(walking(fomdgt3_6),445).
orientation(fomdgt3_1,150,446). coords(fomdgt3_1,46,102,446).  visible(fomdgt3_1,446). happensAt(inactive(fomdgt3_1),446). orientation(fomdgt3_6,160,446). coords(fomdgt3_6,49,103,446).  visible(fomdgt3_6,446). happensAt(walking(fomdgt3_6),446).
orientation(fomdgt3_1,150,447). coords(fomdgt3_1,46,102,447).  visible(fomdgt3_1,447). happensAt(inactive(fomdgt3_1),447). orientation(fomdgt3_6,160,447). coords(fomdgt3_6,49,103,447).  visible(fomdgt3_6,447). happensAt(walking(fomdgt3_6),447).
orientation(fomdgt3_1,150,448). coords(fomdgt3_1,46,102,448).  visible(fomdgt3_1,448). happensAt(inactive(fomdgt3_1),448). orientation(fomdgt3_6,160,448). coords(fomdgt3_6,49,103,448).  visible(fomdgt3_6,448). happensAt(walking(fomdgt3_6),448).
orientation(fomdgt3_1,150,449). coords(fomdgt3_1,46,102,449).  visible(fomdgt3_1,449). happensAt(inactive(fomdgt3_1),449). orientation(fomdgt3_6,160,449). coords(fomdgt3_6,49,103,449).  visible(fomdgt3_6,449). happensAt(walking(fomdgt3_6),449).
orientation(fomdgt3_1,150,450). coords(fomdgt3_1,46,102,450).  visible(fomdgt3_1,450). happensAt(inactive(fomdgt3_1),450). orientation(fomdgt3_6,160,450). coords(fomdgt3_6,49,103,450).  visible(fomdgt3_6,450). happensAt(walking(fomdgt3_6),450).
#show holdsAt/2.