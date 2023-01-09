
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
orientation(bww1gt_1,0,459). coords(bww1gt_1,105,264,459).  visible(bww1gt_1,459). happensAt(inactive(bww1gt_1),459). orientation(bww1gt_2,150,459). coords(bww1gt_2,162,148,459).  visible(bww1gt_2,459). happensAt(walking(bww1gt_2),459).
orientation(bww1gt_1,0,460). coords(bww1gt_1,105,264,460).  visible(bww1gt_1,460). happensAt(inactive(bww1gt_1),460). orientation(bww1gt_2,150,460). coords(bww1gt_2,163,148,460).  visible(bww1gt_2,460). happensAt(walking(bww1gt_2),460).
orientation(bww1gt_1,0,461). coords(bww1gt_1,105,264,461).  visible(bww1gt_1,461). happensAt(inactive(bww1gt_1),461). orientation(bww1gt_2,148,461). coords(bww1gt_2,163,149,461).  visible(bww1gt_2,461). happensAt(walking(bww1gt_2),461).
orientation(bww1gt_1,0,462). coords(bww1gt_1,105,264,462).  visible(bww1gt_1,462). happensAt(inactive(bww1gt_1),462). orientation(bww1gt_2,147,462). coords(bww1gt_2,164,150,462).  visible(bww1gt_2,462). happensAt(walking(bww1gt_2),462).
orientation(bww1gt_1,0,463). coords(bww1gt_1,105,264,463).  visible(bww1gt_1,463). happensAt(inactive(bww1gt_1),463). orientation(bww1gt_2,146,463). coords(bww1gt_2,164,151,463).  visible(bww1gt_2,463). happensAt(walking(bww1gt_2),463).
orientation(bww1gt_1,0,464). coords(bww1gt_1,105,264,464).  visible(bww1gt_1,464). happensAt(inactive(bww1gt_1),464). orientation(bww1gt_2,143,464). coords(bww1gt_2,164,152,464).  visible(bww1gt_2,464). happensAt(walking(bww1gt_2),464).
orientation(bww1gt_1,0,465). coords(bww1gt_1,105,264,465).  visible(bww1gt_1,465). happensAt(inactive(bww1gt_1),465). orientation(bww1gt_2,145,465). coords(bww1gt_2,165,154,465).  visible(bww1gt_2,465). happensAt(walking(bww1gt_2),465).
orientation(bww1gt_1,0,466). coords(bww1gt_1,105,264,466).  visible(bww1gt_1,466). happensAt(inactive(bww1gt_1),466). orientation(bww1gt_2,150,466). coords(bww1gt_2,165,155,466).  visible(bww1gt_2,466). happensAt(walking(bww1gt_2),466).
orientation(bww1gt_1,0,467). coords(bww1gt_1,105,264,467).  visible(bww1gt_1,467). happensAt(inactive(bww1gt_1),467). orientation(bww1gt_2,151,467). coords(bww1gt_2,166,157,467).  visible(bww1gt_2,467). happensAt(walking(bww1gt_2),467).
orientation(bww1gt_1,0,468). coords(bww1gt_1,105,264,468).  visible(bww1gt_1,468). happensAt(inactive(bww1gt_1),468). orientation(bww1gt_2,151,468). coords(bww1gt_2,166,158,468).  visible(bww1gt_2,468). happensAt(walking(bww1gt_2),468).
orientation(bww1gt_1,0,469). coords(bww1gt_1,105,264,469).  visible(bww1gt_1,469). happensAt(inactive(bww1gt_1),469). orientation(bww1gt_2,151,469). coords(bww1gt_2,167,160,469).  visible(bww1gt_2,469). happensAt(walking(bww1gt_2),469).
orientation(bww1gt_1,0,470). coords(bww1gt_1,105,264,470).  visible(bww1gt_1,470). happensAt(inactive(bww1gt_1),470). orientation(bww1gt_2,151,470). coords(bww1gt_2,167,160,470).  visible(bww1gt_2,470). happensAt(walking(bww1gt_2),470).
#show holdsAt/2.