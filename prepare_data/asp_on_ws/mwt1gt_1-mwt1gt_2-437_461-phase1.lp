
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

	orientation(mwt1gt_1,160,437). coords(mwt1gt_1,74,104,437).  visible(mwt1gt_1,437). happensAt(walking(mwt1gt_1),437). orientation(mwt1gt_2,140,437). coords(mwt1gt_2,75,92,437).  visible(mwt1gt_2,437). happensAt(walking(mwt1gt_2),437).
orientation(mwt1gt_1,160,438). coords(mwt1gt_1,74,104,438).  visible(mwt1gt_1,438). happensAt(walking(mwt1gt_1),438). orientation(mwt1gt_2,140,438). coords(mwt1gt_2,75,92,438).  visible(mwt1gt_2,438). happensAt(walking(mwt1gt_2),438).
orientation(mwt1gt_1,160,439). coords(mwt1gt_1,74,104,439).  visible(mwt1gt_1,439). happensAt(walking(mwt1gt_1),439). orientation(mwt1gt_2,140,439). coords(mwt1gt_2,75,92,439).  visible(mwt1gt_2,439). happensAt(walking(mwt1gt_2),439).
orientation(mwt1gt_1,160,440). coords(mwt1gt_1,74,104,440).  visible(mwt1gt_1,440). happensAt(walking(mwt1gt_1),440). orientation(mwt1gt_2,140,440). coords(mwt1gt_2,75,92,440).  visible(mwt1gt_2,440). happensAt(walking(mwt1gt_2),440).
orientation(mwt1gt_1,160,441). coords(mwt1gt_1,74,104,441).  visible(mwt1gt_1,441). happensAt(walking(mwt1gt_1),441). orientation(mwt1gt_2,140,441). coords(mwt1gt_2,75,92,441).  visible(mwt1gt_2,441). happensAt(walking(mwt1gt_2),441).
orientation(mwt1gt_1,160,442). coords(mwt1gt_1,74,104,442).  visible(mwt1gt_1,442). happensAt(walking(mwt1gt_1),442). orientation(mwt1gt_2,140,442). coords(mwt1gt_2,75,92,442).  visible(mwt1gt_2,442). happensAt(walking(mwt1gt_2),442).
orientation(mwt1gt_1,160,443). coords(mwt1gt_1,74,104,443).  visible(mwt1gt_1,443). happensAt(walking(mwt1gt_1),443). orientation(mwt1gt_2,140,443). coords(mwt1gt_2,75,92,443).  visible(mwt1gt_2,443). happensAt(walking(mwt1gt_2),443).
orientation(mwt1gt_1,160,444). coords(mwt1gt_1,74,104,444).  visible(mwt1gt_1,444). happensAt(walking(mwt1gt_1),444). orientation(mwt1gt_2,140,444). coords(mwt1gt_2,75,92,444).  visible(mwt1gt_2,444). happensAt(walking(mwt1gt_2),444).
orientation(mwt1gt_1,160,445). coords(mwt1gt_1,74,101,445).  visible(mwt1gt_1,445). happensAt(walking(mwt1gt_1),445). orientation(mwt1gt_2,140,445). coords(mwt1gt_2,74,90,445).  visible(mwt1gt_2,445). happensAt(walking(mwt1gt_2),445).
orientation(mwt1gt_1,160,446). coords(mwt1gt_1,74,101,446).  visible(mwt1gt_1,446). happensAt(walking(mwt1gt_1),446). orientation(mwt1gt_2,140,446). coords(mwt1gt_2,74,90,446).  visible(mwt1gt_2,446). happensAt(walking(mwt1gt_2),446).
orientation(mwt1gt_1,160,447). coords(mwt1gt_1,72,99,447).  visible(mwt1gt_1,447). happensAt(walking(mwt1gt_1),447). orientation(mwt1gt_2,140,447). coords(mwt1gt_2,72,88,447).  visible(mwt1gt_2,447). happensAt(walking(mwt1gt_2),447).
orientation(mwt1gt_1,160,448). coords(mwt1gt_1,72,99,448).  visible(mwt1gt_1,448). happensAt(walking(mwt1gt_1),448). orientation(mwt1gt_2,140,448). coords(mwt1gt_2,72,88,448).  visible(mwt1gt_2,448). happensAt(walking(mwt1gt_2),448).
orientation(mwt1gt_1,160,449). coords(mwt1gt_1,72,99,449).  visible(mwt1gt_1,449). happensAt(walking(mwt1gt_1),449). orientation(mwt1gt_2,140,449). coords(mwt1gt_2,72,88,449).  visible(mwt1gt_2,449). happensAt(walking(mwt1gt_2),449).
orientation(mwt1gt_1,160,450). coords(mwt1gt_1,72,99,450).  visible(mwt1gt_1,450). happensAt(walking(mwt1gt_1),450). orientation(mwt1gt_2,140,450). coords(mwt1gt_2,72,88,450).  visible(mwt1gt_2,450). happensAt(walking(mwt1gt_2),450).
orientation(mwt1gt_1,160,451). coords(mwt1gt_1,72,99,451).  visible(mwt1gt_1,451). happensAt(walking(mwt1gt_1),451). orientation(mwt1gt_2,140,451). coords(mwt1gt_2,72,88,451).  visible(mwt1gt_2,451). happensAt(walking(mwt1gt_2),451).
orientation(mwt1gt_1,160,452). coords(mwt1gt_1,72,99,452).  visible(mwt1gt_1,452). happensAt(walking(mwt1gt_1),452). orientation(mwt1gt_2,140,452). coords(mwt1gt_2,72,88,452).  visible(mwt1gt_2,452). happensAt(walking(mwt1gt_2),452).
orientation(mwt1gt_1,160,453). coords(mwt1gt_1,72,97,453).  visible(mwt1gt_1,453). happensAt(walking(mwt1gt_1),453). orientation(mwt1gt_2,140,453). coords(mwt1gt_2,72,87,453).  visible(mwt1gt_2,453). happensAt(walking(mwt1gt_2),453).
orientation(mwt1gt_1,160,454). coords(mwt1gt_1,69,97,454).  visible(mwt1gt_1,454). happensAt(walking(mwt1gt_1),454). orientation(mwt1gt_2,140,454). coords(mwt1gt_2,72,87,454).  visible(mwt1gt_2,454). happensAt(walking(mwt1gt_2),454).
orientation(mwt1gt_1,160,455). coords(mwt1gt_1,69,97,455).  visible(mwt1gt_1,455). happensAt(walking(mwt1gt_1),455). orientation(mwt1gt_2,140,455). coords(mwt1gt_2,73,87,455).  visible(mwt1gt_2,455). happensAt(walking(mwt1gt_2),455).
orientation(mwt1gt_1,160,456). coords(mwt1gt_1,69,97,456).  visible(mwt1gt_1,456). happensAt(walking(mwt1gt_1),456). orientation(mwt1gt_2,140,456). coords(mwt1gt_2,73,87,456).  visible(mwt1gt_2,456). happensAt(walking(mwt1gt_2),456).
orientation(mwt1gt_1,160,457). coords(mwt1gt_1,69,97,457).  visible(mwt1gt_1,457). happensAt(walking(mwt1gt_1),457). orientation(mwt1gt_2,140,457). coords(mwt1gt_2,73,87,457).  visible(mwt1gt_2,457). happensAt(walking(mwt1gt_2),457).
orientation(mwt1gt_1,160,458). coords(mwt1gt_1,69,97,458).  visible(mwt1gt_1,458). happensAt(walking(mwt1gt_1),458). orientation(mwt1gt_2,140,458). coords(mwt1gt_2,73,87,458).  visible(mwt1gt_2,458). happensAt(walking(mwt1gt_2),458).
orientation(mwt1gt_1,160,459). coords(mwt1gt_1,69,97,459).  visible(mwt1gt_1,459). happensAt(walking(mwt1gt_1),459). orientation(mwt1gt_2,140,459). coords(mwt1gt_2,73,87,459).  visible(mwt1gt_2,459). happensAt(walking(mwt1gt_2),459).
orientation(mwt1gt_1,160,460). coords(mwt1gt_1,69,97,460).  visible(mwt1gt_1,460). happensAt(walking(mwt1gt_1),460). orientation(mwt1gt_2,140,460). coords(mwt1gt_2,73,87,460).  visible(mwt1gt_2,460). happensAt(walking(mwt1gt_2),460).
orientation(mwt1gt_1,160,461). coords(mwt1gt_1,69,97,461).  visible(mwt1gt_1,461). happensAt(walking(mwt1gt_1),461). orientation(mwt1gt_2,140,461). coords(mwt1gt_2,73,87,461).  visible(mwt1gt_2,461). happensAt(walking(mwt1gt_2),461).
#show holdsAt/2.