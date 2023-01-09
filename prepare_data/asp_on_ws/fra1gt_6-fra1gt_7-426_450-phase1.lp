
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

	orientation(fra1gt_6,177,426). coords(fra1gt_6,71,144,426).  visible(fra1gt_6,426). happensAt(running(fra1gt_6),426). orientation(fra1gt_7,146,426). coords(fra1gt_7,74,81,426).  visible(fra1gt_7,426). happensAt(running(fra1gt_7),426).
orientation(fra1gt_6,5,427). coords(fra1gt_6,70,143,427).  visible(fra1gt_6,427). happensAt(running(fra1gt_6),427). orientation(fra1gt_7,146,427). coords(fra1gt_7,72,81,427).  visible(fra1gt_7,427). happensAt(running(fra1gt_7),427).
orientation(fra1gt_6,5,428). coords(fra1gt_6,69,144,428).  visible(fra1gt_6,428). happensAt(running(fra1gt_6),428). orientation(fra1gt_7,146,428). coords(fra1gt_7,71,80,428).  visible(fra1gt_7,428). happensAt(running(fra1gt_7),428).
orientation(fra1gt_6,5,429). coords(fra1gt_6,68,147,429).  visible(fra1gt_6,429). happensAt(running(fra1gt_6),429). orientation(fra1gt_7,146,429). coords(fra1gt_7,70,81,429).  visible(fra1gt_7,429). happensAt(running(fra1gt_7),429).
orientation(fra1gt_6,5,430). coords(fra1gt_6,66,149,430).  visible(fra1gt_6,430). happensAt(running(fra1gt_6),430). orientation(fra1gt_7,146,430). coords(fra1gt_7,69,81,430).  visible(fra1gt_7,430). happensAt(running(fra1gt_7),430).
orientation(fra1gt_6,5,431). coords(fra1gt_6,65,150,431).  visible(fra1gt_6,431). happensAt(running(fra1gt_6),431). orientation(fra1gt_7,146,431). coords(fra1gt_7,68,81,431).  visible(fra1gt_7,431). happensAt(running(fra1gt_7),431).
orientation(fra1gt_6,5,432). coords(fra1gt_6,63,152,432).  visible(fra1gt_6,432). happensAt(running(fra1gt_6),432). orientation(fra1gt_7,146,432). coords(fra1gt_7,68,82,432).  visible(fra1gt_7,432). happensAt(running(fra1gt_7),432).
orientation(fra1gt_6,5,433). coords(fra1gt_6,62,153,433).  visible(fra1gt_6,433). happensAt(running(fra1gt_6),433). orientation(fra1gt_7,146,433). coords(fra1gt_7,68,82,433).  visible(fra1gt_7,433). happensAt(running(fra1gt_7),433).
orientation(fra1gt_6,5,434). coords(fra1gt_6,61,154,434).  visible(fra1gt_6,434). happensAt(running(fra1gt_6),434). orientation(fra1gt_7,146,434). coords(fra1gt_7,66,82,434).  visible(fra1gt_7,434). happensAt(running(fra1gt_7),434).
orientation(fra1gt_6,177,435). coords(fra1gt_6,59,154,435).  visible(fra1gt_6,435). happensAt(running(fra1gt_6),435). orientation(fra1gt_7,146,435). coords(fra1gt_7,66,83,435).  visible(fra1gt_7,435). happensAt(running(fra1gt_7),435).
orientation(fra1gt_6,3,436). coords(fra1gt_6,58,156,436).  visible(fra1gt_6,436). happensAt(running(fra1gt_6),436). orientation(fra1gt_7,146,436). coords(fra1gt_7,64,83,436).  visible(fra1gt_7,436). happensAt(running(fra1gt_7),436).
orientation(fra1gt_6,4,437). coords(fra1gt_6,58,157,437).  visible(fra1gt_6,437). happensAt(running(fra1gt_6),437). orientation(fra1gt_7,146,437). coords(fra1gt_7,63,84,437).  visible(fra1gt_7,437). happensAt(walking(fra1gt_7),437).
orientation(fra1gt_6,179,438). coords(fra1gt_6,57,160,438).  visible(fra1gt_6,438). happensAt(running(fra1gt_6),438). orientation(fra1gt_7,146,438). coords(fra1gt_7,63,84,438).  visible(fra1gt_7,438). happensAt(walking(fra1gt_7),438).
orientation(fra1gt_6,176,439). coords(fra1gt_6,57,161,439).  visible(fra1gt_6,439). happensAt(running(fra1gt_6),439). orientation(fra1gt_7,146,439). coords(fra1gt_7,63,85,439).  visible(fra1gt_7,439). happensAt(walking(fra1gt_7),439).
orientation(fra1gt_6,166,440). coords(fra1gt_6,56,164,440).  visible(fra1gt_6,440). happensAt(running(fra1gt_6),440). orientation(fra1gt_7,146,440). coords(fra1gt_7,63,85,440).  visible(fra1gt_7,440). happensAt(walking(fra1gt_7),440).
orientation(fra1gt_6,173,441). coords(fra1gt_6,54,166,441).  visible(fra1gt_6,441). happensAt(running(fra1gt_6),441). orientation(fra1gt_7,146,441). coords(fra1gt_7,62,85,441).  visible(fra1gt_7,441). happensAt(walking(fra1gt_7),441).
orientation(fra1gt_6,171,442). coords(fra1gt_6,52,167,442).  visible(fra1gt_6,442). happensAt(running(fra1gt_6),442). orientation(fra1gt_7,146,442). coords(fra1gt_7,61,86,442).  visible(fra1gt_7,442). happensAt(walking(fra1gt_7),442).
orientation(fra1gt_6,167,443). coords(fra1gt_6,51,169,443).  visible(fra1gt_6,443). happensAt(running(fra1gt_6),443). orientation(fra1gt_7,146,443). coords(fra1gt_7,60,87,443).  visible(fra1gt_7,443). happensAt(walking(fra1gt_7),443).
orientation(fra1gt_6,172,444). coords(fra1gt_6,49,170,444).  visible(fra1gt_6,444). happensAt(running(fra1gt_6),444). orientation(fra1gt_7,146,444). coords(fra1gt_7,60,87,444).  visible(fra1gt_7,444). happensAt(walking(fra1gt_7),444).
orientation(fra1gt_6,172,445). coords(fra1gt_6,48,171,445).  visible(fra1gt_6,445). happensAt(running(fra1gt_6),445). orientation(fra1gt_7,146,445). coords(fra1gt_7,60,87,445).  visible(fra1gt_7,445). happensAt(walking(fra1gt_7),445).
orientation(fra1gt_6,177,446). coords(fra1gt_6,47,171,446).  visible(fra1gt_6,446). happensAt(running(fra1gt_6),446). orientation(fra1gt_7,146,446). coords(fra1gt_7,59,87,446).  visible(fra1gt_7,446). happensAt(walking(fra1gt_7),446).
orientation(fra1gt_6,172,447). coords(fra1gt_6,46,172,447).  visible(fra1gt_6,447). happensAt(running(fra1gt_6),447). orientation(fra1gt_7,146,447). coords(fra1gt_7,58,88,447).  visible(fra1gt_7,447). happensAt(walking(fra1gt_7),447).
orientation(fra1gt_6,174,448). coords(fra1gt_6,46,173,448).  visible(fra1gt_6,448). happensAt(running(fra1gt_6),448). orientation(fra1gt_7,146,448). coords(fra1gt_7,58,88,448).  visible(fra1gt_7,448). happensAt(walking(fra1gt_7),448).
orientation(fra1gt_6,174,449). coords(fra1gt_6,45,175,449).  visible(fra1gt_6,449). happensAt(running(fra1gt_6),449). orientation(fra1gt_7,146,449). coords(fra1gt_7,58,88,449).  visible(fra1gt_7,449). happensAt(walking(fra1gt_7),449).
orientation(fra1gt_6,166,450). coords(fra1gt_6,45,175,450).  visible(fra1gt_6,450). happensAt(running(fra1gt_6),450). orientation(fra1gt_7,146,450). coords(fra1gt_7,57,87,450).  visible(fra1gt_7,450). happensAt(walking(fra1gt_7),450).
#show holdsAt/2.