
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

	orientation(bww1gt_1,178,434). coords(bww1gt_1,105,264,434).  visible(bww1gt_1,434). happensAt(inactive(bww1gt_1),434). orientation(bww1gt_2,151,434). coords(bww1gt_2,142,146,434).  visible(bww1gt_2,434). happensAt(walking(bww1gt_2),434).
orientation(bww1gt_1,0,435). coords(bww1gt_1,105,264,435).  visible(bww1gt_1,435). happensAt(inactive(bww1gt_1),435). orientation(bww1gt_2,153,435). coords(bww1gt_2,144,146,435).  visible(bww1gt_2,435). happensAt(walking(bww1gt_2),435).
orientation(bww1gt_1,0,436). coords(bww1gt_1,105,264,436).  visible(bww1gt_1,436). happensAt(inactive(bww1gt_1),436). orientation(bww1gt_2,153,436). coords(bww1gt_2,144,146,436).  visible(bww1gt_2,436). happensAt(walking(bww1gt_2),436).
orientation(bww1gt_1,0,437). coords(bww1gt_1,105,264,437).  visible(bww1gt_1,437). happensAt(inactive(bww1gt_1),437). orientation(bww1gt_2,156,437). coords(bww1gt_2,145,146,437).  visible(bww1gt_2,437). happensAt(walking(bww1gt_2),437).
orientation(bww1gt_1,0,438). coords(bww1gt_1,105,264,438).  visible(bww1gt_1,438). happensAt(inactive(bww1gt_1),438). orientation(bww1gt_2,156,438). coords(bww1gt_2,146,146,438).  visible(bww1gt_2,438). happensAt(walking(bww1gt_2),438).
orientation(bww1gt_1,0,439). coords(bww1gt_1,105,264,439).  visible(bww1gt_1,439). happensAt(inactive(bww1gt_1),439). orientation(bww1gt_2,155,439). coords(bww1gt_2,146,147,439).  visible(bww1gt_2,439). happensAt(walking(bww1gt_2),439).
orientation(bww1gt_1,0,440). coords(bww1gt_1,105,264,440).  visible(bww1gt_1,440). happensAt(inactive(bww1gt_1),440). orientation(bww1gt_2,153,440). coords(bww1gt_2,146,147,440).  visible(bww1gt_2,440). happensAt(walking(bww1gt_2),440).
orientation(bww1gt_1,0,441). coords(bww1gt_1,105,264,441).  visible(bww1gt_1,441). happensAt(inactive(bww1gt_1),441). orientation(bww1gt_2,155,441). coords(bww1gt_2,146,147,441).  visible(bww1gt_2,441). happensAt(walking(bww1gt_2),441).
orientation(bww1gt_1,0,442). coords(bww1gt_1,105,264,442).  visible(bww1gt_1,442). happensAt(inactive(bww1gt_1),442). orientation(bww1gt_2,153,442). coords(bww1gt_2,146,147,442).  visible(bww1gt_2,442). happensAt(walking(bww1gt_2),442).
orientation(bww1gt_1,0,443). coords(bww1gt_1,105,264,443).  visible(bww1gt_1,443). happensAt(inactive(bww1gt_1),443). orientation(bww1gt_2,152,443). coords(bww1gt_2,148,147,443).  visible(bww1gt_2,443). happensAt(walking(bww1gt_2),443).
orientation(bww1gt_1,0,444). coords(bww1gt_1,105,264,444).  visible(bww1gt_1,444). happensAt(inactive(bww1gt_1),444). orientation(bww1gt_2,152,444). coords(bww1gt_2,148,147,444).  visible(bww1gt_2,444). happensAt(walking(bww1gt_2),444).
orientation(bww1gt_1,0,445). coords(bww1gt_1,105,264,445).  visible(bww1gt_1,445). happensAt(inactive(bww1gt_1),445). orientation(bww1gt_2,153,445). coords(bww1gt_2,148,147,445).  visible(bww1gt_2,445). happensAt(walking(bww1gt_2),445).
orientation(bww1gt_1,0,446). coords(bww1gt_1,105,264,446).  visible(bww1gt_1,446). happensAt(inactive(bww1gt_1),446). orientation(bww1gt_2,155,446). coords(bww1gt_2,151,147,446).  visible(bww1gt_2,446). happensAt(walking(bww1gt_2),446).
orientation(bww1gt_1,0,447). coords(bww1gt_1,105,264,447).  visible(bww1gt_1,447). happensAt(inactive(bww1gt_1),447). orientation(bww1gt_2,155,447). coords(bww1gt_2,153,147,447).  visible(bww1gt_2,447). happensAt(walking(bww1gt_2),447).
orientation(bww1gt_1,0,448). coords(bww1gt_1,105,264,448).  visible(bww1gt_1,448). happensAt(inactive(bww1gt_1),448). orientation(bww1gt_2,154,448). coords(bww1gt_2,155,148,448).  visible(bww1gt_2,448). happensAt(walking(bww1gt_2),448).
orientation(bww1gt_1,0,449). coords(bww1gt_1,105,264,449).  visible(bww1gt_1,449). happensAt(inactive(bww1gt_1),449). orientation(bww1gt_2,154,449). coords(bww1gt_2,156,148,449).  visible(bww1gt_2,449). happensAt(walking(bww1gt_2),449).
orientation(bww1gt_1,0,450). coords(bww1gt_1,105,264,450).  visible(bww1gt_1,450). happensAt(inactive(bww1gt_1),450). orientation(bww1gt_2,153,450). coords(bww1gt_2,158,148,450).  visible(bww1gt_2,450). happensAt(walking(bww1gt_2),450).
orientation(bww1gt_1,0,451). coords(bww1gt_1,105,264,451).  visible(bww1gt_1,451). happensAt(inactive(bww1gt_1),451). orientation(bww1gt_2,152,451). coords(bww1gt_2,159,149,451).  visible(bww1gt_2,451). happensAt(walking(bww1gt_2),451).
orientation(bww1gt_1,0,452). coords(bww1gt_1,105,264,452).  visible(bww1gt_1,452). happensAt(inactive(bww1gt_1),452). orientation(bww1gt_2,152,452). coords(bww1gt_2,160,149,452).  visible(bww1gt_2,452). happensAt(walking(bww1gt_2),452).
orientation(bww1gt_1,0,453). coords(bww1gt_1,105,264,453).  visible(bww1gt_1,453). happensAt(inactive(bww1gt_1),453). orientation(bww1gt_2,153,453). coords(bww1gt_2,160,149,453).  visible(bww1gt_2,453). happensAt(walking(bww1gt_2),453).
orientation(bww1gt_1,0,454). coords(bww1gt_1,105,264,454).  visible(bww1gt_1,454). happensAt(inactive(bww1gt_1),454). orientation(bww1gt_2,154,454). coords(bww1gt_2,161,149,454).  visible(bww1gt_2,454). happensAt(walking(bww1gt_2),454).
orientation(bww1gt_1,0,455). coords(bww1gt_1,105,264,455).  visible(bww1gt_1,455). happensAt(inactive(bww1gt_1),455). orientation(bww1gt_2,153,455). coords(bww1gt_2,161,149,455).  visible(bww1gt_2,455). happensAt(walking(bww1gt_2),455).
orientation(bww1gt_1,0,456). coords(bww1gt_1,105,264,456).  visible(bww1gt_1,456). happensAt(inactive(bww1gt_1),456). orientation(bww1gt_2,153,456). coords(bww1gt_2,162,148,456).  visible(bww1gt_2,456). happensAt(walking(bww1gt_2),456).
orientation(bww1gt_1,0,457). coords(bww1gt_1,105,264,457).  visible(bww1gt_1,457). happensAt(inactive(bww1gt_1),457). orientation(bww1gt_2,152,457). coords(bww1gt_2,162,148,457).  visible(bww1gt_2,457). happensAt(walking(bww1gt_2),457).
orientation(bww1gt_1,0,458). coords(bww1gt_1,105,264,458).  visible(bww1gt_1,458). happensAt(inactive(bww1gt_1),458). orientation(bww1gt_2,151,458). coords(bww1gt_2,162,148,458).  visible(bww1gt_2,458). happensAt(walking(bww1gt_2),458).
#show holdsAt/2.