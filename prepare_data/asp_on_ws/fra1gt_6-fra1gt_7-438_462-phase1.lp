
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
orientation(fra1gt_6,166,451). coords(fra1gt_6,44,176,451).  visible(fra1gt_6,451). happensAt(running(fra1gt_6),451). orientation(fra1gt_7,146,451). coords(fra1gt_7,57,88,451).  visible(fra1gt_7,451). happensAt(walking(fra1gt_7),451).
orientation(fra1gt_6,166,452). coords(fra1gt_6,44,179,452).  visible(fra1gt_6,452). happensAt(running(fra1gt_6),452). orientation(fra1gt_7,146,452). coords(fra1gt_7,56,88,452).  visible(fra1gt_7,452). happensAt(walking(fra1gt_7),452).
orientation(fra1gt_6,166,453). coords(fra1gt_6,44,180,453).  visible(fra1gt_6,453). happensAt(running(fra1gt_6),453). orientation(fra1gt_7,146,453). coords(fra1gt_7,55,88,453).  visible(fra1gt_7,453). happensAt(walking(fra1gt_7),453).
orientation(fra1gt_6,166,454). coords(fra1gt_6,42,182,454).  visible(fra1gt_6,454). happensAt(running(fra1gt_6),454). orientation(fra1gt_7,146,454). coords(fra1gt_7,55,88,454).  visible(fra1gt_7,454). happensAt(walking(fra1gt_7),454).
orientation(fra1gt_6,166,455). coords(fra1gt_6,42,182,455).  visible(fra1gt_6,455). happensAt(running(fra1gt_6),455). orientation(fra1gt_7,146,455). coords(fra1gt_7,55,89,455).  visible(fra1gt_7,455). happensAt(walking(fra1gt_7),455).
orientation(fra1gt_6,166,456). coords(fra1gt_6,41,183,456).  visible(fra1gt_6,456). happensAt(running(fra1gt_6),456). orientation(fra1gt_7,146,456). coords(fra1gt_7,54,90,456).  visible(fra1gt_7,456). happensAt(walking(fra1gt_7),456).
orientation(fra1gt_6,166,457). coords(fra1gt_6,40,185,457).  visible(fra1gt_6,457). happensAt(running(fra1gt_6),457). orientation(fra1gt_7,146,457). coords(fra1gt_7,54,90,457).  visible(fra1gt_7,457). happensAt(walking(fra1gt_7),457).
orientation(fra1gt_6,166,458). coords(fra1gt_6,39,186,458).  visible(fra1gt_6,458). happensAt(running(fra1gt_6),458). orientation(fra1gt_7,146,458). coords(fra1gt_7,54,90,458).  visible(fra1gt_7,458). happensAt(walking(fra1gt_7),458).
orientation(fra1gt_6,166,459). coords(fra1gt_6,39,188,459).  visible(fra1gt_6,459). happensAt(running(fra1gt_6),459). orientation(fra1gt_7,146,459). coords(fra1gt_7,54,90,459).  visible(fra1gt_7,459). happensAt(walking(fra1gt_7),459).
orientation(fra1gt_6,166,460). coords(fra1gt_6,38,189,460).  visible(fra1gt_6,460). happensAt(running(fra1gt_6),460). orientation(fra1gt_7,146,460). coords(fra1gt_7,54,90,460).  visible(fra1gt_7,460). happensAt(walking(fra1gt_7),460).
orientation(fra1gt_6,166,461). coords(fra1gt_6,38,190,461).  visible(fra1gt_6,461). happensAt(running(fra1gt_6),461). orientation(fra1gt_7,146,461). coords(fra1gt_7,53,90,461).  visible(fra1gt_7,461). happensAt(walking(fra1gt_7),461).
orientation(fra1gt_6,166,462). coords(fra1gt_6,38,191,462).  visible(fra1gt_6,462). happensAt(running(fra1gt_6),462). orientation(fra1gt_7,146,462). coords(fra1gt_7,53,90,462).  visible(fra1gt_7,462). happensAt(walking(fra1gt_7),462).
#show holdsAt/2.