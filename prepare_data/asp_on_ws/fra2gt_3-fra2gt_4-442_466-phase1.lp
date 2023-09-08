
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
orientation(fra2gt_3,153,455). coords(fra2gt_3,42,163,455).  visible(fra2gt_3,455). happensAt(walking(fra2gt_3),455). orientation(fra2gt_4,150,455). coords(fra2gt_4,72,81,455).  visible(fra2gt_4,455). happensAt(running(fra2gt_4),455).
orientation(fra2gt_3,153,456). coords(fra2gt_3,41,163,456).  visible(fra2gt_3,456). happensAt(walking(fra2gt_3),456). orientation(fra2gt_4,150,456). coords(fra2gt_4,72,81,456).  visible(fra2gt_4,456). happensAt(running(fra2gt_4),456).
orientation(fra2gt_3,153,457). coords(fra2gt_3,41,163,457).  visible(fra2gt_3,457). happensAt(walking(fra2gt_3),457). orientation(fra2gt_4,150,457). coords(fra2gt_4,71,80,457).  visible(fra2gt_4,457). happensAt(running(fra2gt_4),457).
orientation(fra2gt_3,153,458). coords(fra2gt_3,41,163,458).  visible(fra2gt_3,458). happensAt(walking(fra2gt_3),458). orientation(fra2gt_4,150,458). coords(fra2gt_4,71,80,458).  visible(fra2gt_4,458). happensAt(running(fra2gt_4),458).
orientation(fra2gt_3,153,459). coords(fra2gt_3,41,163,459).  visible(fra2gt_3,459). happensAt(walking(fra2gt_3),459). orientation(fra2gt_4,150,459). coords(fra2gt_4,71,80,459).  visible(fra2gt_4,459). happensAt(running(fra2gt_4),459).
orientation(fra2gt_3,153,460). coords(fra2gt_3,41,163,460).  visible(fra2gt_3,460). happensAt(walking(fra2gt_3),460). orientation(fra2gt_4,150,460). coords(fra2gt_4,70,80,460).  visible(fra2gt_4,460). happensAt(running(fra2gt_4),460).
orientation(fra2gt_3,153,461). coords(fra2gt_3,40,162,461).  visible(fra2gt_3,461). happensAt(walking(fra2gt_3),461). orientation(fra2gt_4,150,461). coords(fra2gt_4,70,80,461).  visible(fra2gt_4,461). happensAt(running(fra2gt_4),461).
orientation(fra2gt_3,153,462). coords(fra2gt_3,40,162,462).  visible(fra2gt_3,462). happensAt(walking(fra2gt_3),462). orientation(fra2gt_4,150,462). coords(fra2gt_4,69,81,462).  visible(fra2gt_4,462). happensAt(walking(fra2gt_4),462).
orientation(fra2gt_3,153,463). coords(fra2gt_3,39,162,463).  visible(fra2gt_3,463). happensAt(walking(fra2gt_3),463). orientation(fra2gt_4,150,463). coords(fra2gt_4,69,81,463).  visible(fra2gt_4,463). happensAt(walking(fra2gt_4),463).
orientation(fra2gt_3,153,464). coords(fra2gt_3,39,161,464).  visible(fra2gt_3,464). happensAt(walking(fra2gt_3),464). orientation(fra2gt_4,150,464). coords(fra2gt_4,69,81,464).  visible(fra2gt_4,464). happensAt(walking(fra2gt_4),464).
orientation(fra2gt_3,153,465). coords(fra2gt_3,39,161,465).  visible(fra2gt_3,465). happensAt(walking(fra2gt_3),465). orientation(fra2gt_4,150,465). coords(fra2gt_4,69,81,465).  visible(fra2gt_4,465). happensAt(walking(fra2gt_4),465).
orientation(fra2gt_3,153,466). coords(fra2gt_3,38,161,466).  visible(fra2gt_3,466). happensAt(walking(fra2gt_3),466). orientation(fra2gt_4,150,466). coords(fra2gt_4,68,81,466).  visible(fra2gt_4,466). happensAt(walking(fra2gt_4),466).
#show holdsAt/2.