
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

	orientation(wk2gt_1,125,732). coords(wk2gt_1,114,53,732).  visible(wk2gt_1,732). happensAt(inactive(wk2gt_1),732). orientation(wk2gt_2,142,732). coords(wk2gt_2,99,63,732).  visible(wk2gt_2,732). happensAt(inactive(wk2gt_2),732).
orientation(wk2gt_1,125,733). coords(wk2gt_1,114,53,733).  visible(wk2gt_1,733). happensAt(inactive(wk2gt_1),733). orientation(wk2gt_2,142,733). coords(wk2gt_2,99,63,733).  visible(wk2gt_2,733). happensAt(inactive(wk2gt_2),733).
orientation(wk2gt_1,125,734). coords(wk2gt_1,114,53,734).  visible(wk2gt_1,734). happensAt(inactive(wk2gt_1),734). orientation(wk2gt_2,142,734). coords(wk2gt_2,99,63,734).  visible(wk2gt_2,734). happensAt(inactive(wk2gt_2),734).
orientation(wk2gt_1,125,735). coords(wk2gt_1,114,53,735).  visible(wk2gt_1,735). happensAt(inactive(wk2gt_1),735). orientation(wk2gt_2,142,735). coords(wk2gt_2,99,63,735).  visible(wk2gt_2,735). happensAt(inactive(wk2gt_2),735).
orientation(wk2gt_1,125,736). coords(wk2gt_1,114,53,736).  visible(wk2gt_1,736). happensAt(inactive(wk2gt_1),736). orientation(wk2gt_2,142,736). coords(wk2gt_2,99,63,736).  visible(wk2gt_2,736). happensAt(inactive(wk2gt_2),736).
orientation(wk2gt_1,125,737). coords(wk2gt_1,114,53,737).  visible(wk2gt_1,737). happensAt(inactive(wk2gt_1),737). orientation(wk2gt_2,142,737). coords(wk2gt_2,99,63,737).  visible(wk2gt_2,737). happensAt(inactive(wk2gt_2),737).
orientation(wk2gt_1,125,738). coords(wk2gt_1,114,53,738).  visible(wk2gt_1,738). happensAt(inactive(wk2gt_1),738). orientation(wk2gt_2,142,738). coords(wk2gt_2,99,63,738).  visible(wk2gt_2,738). happensAt(inactive(wk2gt_2),738).
orientation(wk2gt_1,125,739). coords(wk2gt_1,114,53,739).  visible(wk2gt_1,739). happensAt(inactive(wk2gt_1),739). orientation(wk2gt_2,142,739). coords(wk2gt_2,99,63,739).  visible(wk2gt_2,739). happensAt(inactive(wk2gt_2),739).
orientation(wk2gt_1,125,740). coords(wk2gt_1,114,53,740).  visible(wk2gt_1,740). happensAt(inactive(wk2gt_1),740). orientation(wk2gt_2,142,740). coords(wk2gt_2,99,63,740).  visible(wk2gt_2,740). happensAt(inactive(wk2gt_2),740).
orientation(wk2gt_1,125,741). coords(wk2gt_1,114,53,741).  visible(wk2gt_1,741). happensAt(inactive(wk2gt_1),741). orientation(wk2gt_2,142,741). coords(wk2gt_2,99,63,741).  visible(wk2gt_2,741). happensAt(inactive(wk2gt_2),741).
orientation(wk2gt_1,125,742). coords(wk2gt_1,114,53,742).  visible(wk2gt_1,742). happensAt(inactive(wk2gt_1),742). orientation(wk2gt_2,142,742). coords(wk2gt_2,99,63,742).  visible(wk2gt_2,742). happensAt(inactive(wk2gt_2),742).
orientation(wk2gt_1,125,743). coords(wk2gt_1,114,53,743).  visible(wk2gt_1,743). happensAt(inactive(wk2gt_1),743). orientation(wk2gt_2,142,743). coords(wk2gt_2,99,63,743).  visible(wk2gt_2,743). happensAt(inactive(wk2gt_2),743).
orientation(wk2gt_1,125,744). coords(wk2gt_1,114,53,744).  visible(wk2gt_1,744). happensAt(inactive(wk2gt_1),744). orientation(wk2gt_2,142,744). coords(wk2gt_2,99,63,744).  visible(wk2gt_2,744). happensAt(inactive(wk2gt_2),744).
orientation(wk2gt_1,125,745). coords(wk2gt_1,115,53,745).  visible(wk2gt_1,745). happensAt(inactive(wk2gt_1),745). orientation(wk2gt_2,142,745). coords(wk2gt_2,99,63,745).  visible(wk2gt_2,745). happensAt(inactive(wk2gt_2),745).
orientation(wk2gt_1,125,746). coords(wk2gt_1,115,53,746).  visible(wk2gt_1,746). happensAt(inactive(wk2gt_1),746). orientation(wk2gt_2,142,746). coords(wk2gt_2,99,63,746).  visible(wk2gt_2,746). happensAt(inactive(wk2gt_2),746).
orientation(wk2gt_1,125,747). coords(wk2gt_1,115,53,747).  visible(wk2gt_1,747). happensAt(inactive(wk2gt_1),747). orientation(wk2gt_2,142,747). coords(wk2gt_2,99,63,747).  visible(wk2gt_2,747). happensAt(inactive(wk2gt_2),747).
orientation(wk2gt_1,125,748). coords(wk2gt_1,115,53,748).  visible(wk2gt_1,748). happensAt(inactive(wk2gt_1),748). orientation(wk2gt_2,142,748). coords(wk2gt_2,99,63,748).  visible(wk2gt_2,748). happensAt(inactive(wk2gt_2),748).
orientation(wk2gt_1,125,749). coords(wk2gt_1,115,53,749).  visible(wk2gt_1,749). happensAt(inactive(wk2gt_1),749). orientation(wk2gt_2,142,749). coords(wk2gt_2,99,63,749).  visible(wk2gt_2,749). happensAt(inactive(wk2gt_2),749).
orientation(wk2gt_1,125,750). coords(wk2gt_1,115,53,750).  visible(wk2gt_1,750). happensAt(inactive(wk2gt_1),750). orientation(wk2gt_2,142,750). coords(wk2gt_2,99,63,750).  visible(wk2gt_2,750). happensAt(inactive(wk2gt_2),750).
orientation(wk2gt_1,125,751). coords(wk2gt_1,115,53,751).  visible(wk2gt_1,751). happensAt(inactive(wk2gt_1),751). orientation(wk2gt_2,142,751). coords(wk2gt_2,99,63,751).  visible(wk2gt_2,751). happensAt(inactive(wk2gt_2),751).
orientation(wk2gt_1,125,752). coords(wk2gt_1,115,53,752).  visible(wk2gt_1,752). happensAt(inactive(wk2gt_1),752). orientation(wk2gt_2,142,752). coords(wk2gt_2,99,63,752).  visible(wk2gt_2,752). happensAt(inactive(wk2gt_2),752).
orientation(wk2gt_1,125,753). coords(wk2gt_1,115,53,753).  visible(wk2gt_1,753). happensAt(inactive(wk2gt_1),753). orientation(wk2gt_2,142,753). coords(wk2gt_2,99,63,753).  visible(wk2gt_2,753). happensAt(inactive(wk2gt_2),753).
orientation(wk2gt_1,125,754). coords(wk2gt_1,115,53,754).  visible(wk2gt_1,754). happensAt(inactive(wk2gt_1),754). orientation(wk2gt_2,142,754). coords(wk2gt_2,99,63,754).  visible(wk2gt_2,754). happensAt(inactive(wk2gt_2),754).
orientation(wk2gt_1,125,755). coords(wk2gt_1,115,53,755).  visible(wk2gt_1,755). happensAt(inactive(wk2gt_1),755). orientation(wk2gt_2,142,755). coords(wk2gt_2,99,63,755).  visible(wk2gt_2,755). happensAt(inactive(wk2gt_2),755).
orientation(wk2gt_1,125,756). coords(wk2gt_1,115,53,756).  visible(wk2gt_1,756). happensAt(inactive(wk2gt_1),756). orientation(wk2gt_2,142,756). coords(wk2gt_2,99,63,756).  visible(wk2gt_2,756). happensAt(inactive(wk2gt_2),756).
#show holdsAt/2.