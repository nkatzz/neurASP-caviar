
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
orientation(br1gt_2,12,455). coords(br1gt_2,86,208,455).  visible(br1gt_2,455). happensAt(inactive(br1gt_2),455). orientation(br1gt_3,160,455). coords(br1gt_3,70,146,455).  visible(br1gt_3,455). happensAt(walking(br1gt_3),455).
orientation(br1gt_2,12,456). coords(br1gt_2,86,208,456).  visible(br1gt_2,456). happensAt(inactive(br1gt_2),456). orientation(br1gt_3,160,456). coords(br1gt_3,71,146,456).  visible(br1gt_3,456). happensAt(walking(br1gt_3),456).
orientation(br1gt_2,12,457). coords(br1gt_2,86,208,457).  visible(br1gt_2,457). happensAt(inactive(br1gt_2),457). orientation(br1gt_3,160,457). coords(br1gt_3,71,146,457).  visible(br1gt_3,457). happensAt(walking(br1gt_3),457).
orientation(br1gt_2,12,458). coords(br1gt_2,86,208,458).  visible(br1gt_2,458). happensAt(inactive(br1gt_2),458). orientation(br1gt_3,160,458). coords(br1gt_3,71,146,458).  visible(br1gt_3,458). happensAt(walking(br1gt_3),458).
orientation(br1gt_2,12,459). coords(br1gt_2,86,208,459).  visible(br1gt_2,459). happensAt(inactive(br1gt_2),459). orientation(br1gt_3,160,459). coords(br1gt_3,72,146,459).  visible(br1gt_3,459). happensAt(walking(br1gt_3),459).
orientation(br1gt_2,12,460). coords(br1gt_2,86,208,460).  visible(br1gt_2,460). happensAt(inactive(br1gt_2),460). orientation(br1gt_3,160,460). coords(br1gt_3,74,146,460).  visible(br1gt_3,460). happensAt(walking(br1gt_3),460).
orientation(br1gt_2,12,461). coords(br1gt_2,86,208,461).  visible(br1gt_2,461). happensAt(inactive(br1gt_2),461). orientation(br1gt_3,160,461). coords(br1gt_3,76,146,461).  visible(br1gt_3,461). happensAt(walking(br1gt_3),461).
orientation(br1gt_2,12,462). coords(br1gt_2,86,208,462).  visible(br1gt_2,462). happensAt(inactive(br1gt_2),462). orientation(br1gt_3,160,462). coords(br1gt_3,77,146,462).  visible(br1gt_3,462). happensAt(walking(br1gt_3),462).
orientation(br1gt_2,12,463). coords(br1gt_2,86,208,463).  visible(br1gt_2,463). happensAt(inactive(br1gt_2),463). orientation(br1gt_3,160,463). coords(br1gt_3,78,146,463).  visible(br1gt_3,463). happensAt(walking(br1gt_3),463).
orientation(br1gt_2,12,464). coords(br1gt_2,86,208,464).  visible(br1gt_2,464). happensAt(inactive(br1gt_2),464). orientation(br1gt_3,160,464). coords(br1gt_3,78,145,464).  visible(br1gt_3,464). happensAt(walking(br1gt_3),464).
orientation(br1gt_2,12,465). coords(br1gt_2,86,208,465).  visible(br1gt_2,465). happensAt(inactive(br1gt_2),465). orientation(br1gt_3,160,465). coords(br1gt_3,79,145,465).  visible(br1gt_3,465). happensAt(walking(br1gt_3),465).
orientation(br1gt_2,12,466). coords(br1gt_2,86,208,466).  visible(br1gt_2,466). happensAt(inactive(br1gt_2),466). orientation(br1gt_3,160,466). coords(br1gt_3,81,145,466).  visible(br1gt_3,466). happensAt(walking(br1gt_3),466).
#show holdsAt/2.