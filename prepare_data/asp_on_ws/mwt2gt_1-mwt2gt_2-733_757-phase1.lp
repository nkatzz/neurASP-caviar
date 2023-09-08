
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

	orientation(mwt2gt_1,153,733). coords(mwt2gt_1,30,122,733).  visible(mwt2gt_1,733). happensAt(active(mwt2gt_1),733). orientation(mwt2gt_2,152,733). coords(mwt2gt_2,26,130,733).  visible(mwt2gt_2,733). happensAt(active(mwt2gt_2),733).
orientation(mwt2gt_1,153,734). coords(mwt2gt_1,30,121,734).  visible(mwt2gt_1,734). happensAt(active(mwt2gt_1),734). orientation(mwt2gt_2,152,734). coords(mwt2gt_2,27,129,734).  visible(mwt2gt_2,734). happensAt(active(mwt2gt_2),734).
orientation(mwt2gt_1,153,735). coords(mwt2gt_1,30,121,735).  visible(mwt2gt_1,735). happensAt(active(mwt2gt_1),735). orientation(mwt2gt_2,152,735). coords(mwt2gt_2,27,129,735).  visible(mwt2gt_2,735). happensAt(active(mwt2gt_2),735).
orientation(mwt2gt_1,153,736). coords(mwt2gt_1,30,121,736).  visible(mwt2gt_1,736). happensAt(active(mwt2gt_1),736). orientation(mwt2gt_2,152,736). coords(mwt2gt_2,27,129,736).  visible(mwt2gt_2,736). happensAt(active(mwt2gt_2),736).
orientation(mwt2gt_1,153,737). coords(mwt2gt_1,30,121,737).  visible(mwt2gt_1,737). happensAt(active(mwt2gt_1),737). orientation(mwt2gt_2,152,737). coords(mwt2gt_2,27,129,737).  visible(mwt2gt_2,737). happensAt(active(mwt2gt_2),737).
orientation(mwt2gt_1,153,738). coords(mwt2gt_1,30,121,738).  visible(mwt2gt_1,738). happensAt(active(mwt2gt_1),738). orientation(mwt2gt_2,152,738). coords(mwt2gt_2,27,129,738).  visible(mwt2gt_2,738). happensAt(active(mwt2gt_2),738).
orientation(mwt2gt_1,153,739). coords(mwt2gt_1,30,119,739).  visible(mwt2gt_1,739). happensAt(active(mwt2gt_1),739). orientation(mwt2gt_2,152,739). coords(mwt2gt_2,27,129,739).  visible(mwt2gt_2,739). happensAt(active(mwt2gt_2),739).
orientation(mwt2gt_1,153,740). coords(mwt2gt_1,30,119,740).  visible(mwt2gt_1,740). happensAt(active(mwt2gt_1),740). orientation(mwt2gt_2,152,740). coords(mwt2gt_2,27,129,740).  visible(mwt2gt_2,740). happensAt(active(mwt2gt_2),740).
orientation(mwt2gt_1,153,741). coords(mwt2gt_1,30,121,741).  visible(mwt2gt_1,741). happensAt(active(mwt2gt_1),741). orientation(mwt2gt_2,152,741). coords(mwt2gt_2,27,129,741).  visible(mwt2gt_2,741). happensAt(active(mwt2gt_2),741).
orientation(mwt2gt_1,153,742). coords(mwt2gt_1,30,119,742).  visible(mwt2gt_1,742). happensAt(active(mwt2gt_1),742). orientation(mwt2gt_2,152,742). coords(mwt2gt_2,27,129,742).  visible(mwt2gt_2,742). happensAt(active(mwt2gt_2),742).
orientation(mwt2gt_1,153,743). coords(mwt2gt_1,30,119,743).  visible(mwt2gt_1,743). happensAt(active(mwt2gt_1),743). orientation(mwt2gt_2,152,743). coords(mwt2gt_2,27,129,743).  visible(mwt2gt_2,743). happensAt(active(mwt2gt_2),743).
orientation(mwt2gt_1,153,744). coords(mwt2gt_1,30,121,744).  visible(mwt2gt_1,744). happensAt(active(mwt2gt_1),744). orientation(mwt2gt_2,152,744). coords(mwt2gt_2,27,129,744).  visible(mwt2gt_2,744). happensAt(active(mwt2gt_2),744).
orientation(mwt2gt_1,153,745). coords(mwt2gt_1,30,121,745).  visible(mwt2gt_1,745). happensAt(active(mwt2gt_1),745). orientation(mwt2gt_2,152,745). coords(mwt2gt_2,27,128,745).  visible(mwt2gt_2,745). happensAt(active(mwt2gt_2),745).
orientation(mwt2gt_1,153,746). coords(mwt2gt_1,30,121,746).  visible(mwt2gt_1,746). happensAt(active(mwt2gt_1),746). orientation(mwt2gt_2,152,746). coords(mwt2gt_2,27,128,746).  visible(mwt2gt_2,746). happensAt(active(mwt2gt_2),746).
orientation(mwt2gt_1,153,747). coords(mwt2gt_1,30,121,747).  visible(mwt2gt_1,747). happensAt(active(mwt2gt_1),747). orientation(mwt2gt_2,152,747). coords(mwt2gt_2,27,128,747).  visible(mwt2gt_2,747). happensAt(active(mwt2gt_2),747).
orientation(mwt2gt_1,153,748). coords(mwt2gt_1,30,121,748).  visible(mwt2gt_1,748). happensAt(active(mwt2gt_1),748). orientation(mwt2gt_2,152,748). coords(mwt2gt_2,27,128,748).  visible(mwt2gt_2,748). happensAt(active(mwt2gt_2),748).
orientation(mwt2gt_1,153,749). coords(mwt2gt_1,30,121,749).  visible(mwt2gt_1,749). happensAt(active(mwt2gt_1),749). orientation(mwt2gt_2,152,749). coords(mwt2gt_2,27,128,749).  visible(mwt2gt_2,749). happensAt(active(mwt2gt_2),749).
orientation(mwt2gt_1,153,750). coords(mwt2gt_1,30,121,750).  visible(mwt2gt_1,750). happensAt(active(mwt2gt_1),750). orientation(mwt2gt_2,152,750). coords(mwt2gt_2,27,128,750).  visible(mwt2gt_2,750). happensAt(active(mwt2gt_2),750).
orientation(mwt2gt_1,153,751). coords(mwt2gt_1,30,121,751).  visible(mwt2gt_1,751). happensAt(active(mwt2gt_1),751). orientation(mwt2gt_2,152,751). coords(mwt2gt_2,28,128,751).  visible(mwt2gt_2,751). happensAt(active(mwt2gt_2),751).
orientation(mwt2gt_1,153,752). coords(mwt2gt_1,30,121,752).  visible(mwt2gt_1,752). happensAt(active(mwt2gt_1),752). orientation(mwt2gt_2,152,752). coords(mwt2gt_2,28,128,752).  visible(mwt2gt_2,752). happensAt(active(mwt2gt_2),752).
orientation(mwt2gt_1,153,753). coords(mwt2gt_1,30,120,753).  visible(mwt2gt_1,753). happensAt(active(mwt2gt_1),753). orientation(mwt2gt_2,152,753). coords(mwt2gt_2,28,128,753).  visible(mwt2gt_2,753). happensAt(active(mwt2gt_2),753).
orientation(mwt2gt_1,153,754). coords(mwt2gt_1,30,120,754).  visible(mwt2gt_1,754). happensAt(active(mwt2gt_1),754). orientation(mwt2gt_2,152,754). coords(mwt2gt_2,28,128,754).  visible(mwt2gt_2,754). happensAt(active(mwt2gt_2),754).
orientation(mwt2gt_1,153,755). coords(mwt2gt_1,30,120,755).  visible(mwt2gt_1,755). happensAt(active(mwt2gt_1),755). orientation(mwt2gt_2,152,755). coords(mwt2gt_2,28,128,755).  visible(mwt2gt_2,755). happensAt(active(mwt2gt_2),755).
orientation(mwt2gt_1,153,756). coords(mwt2gt_1,30,120,756).  visible(mwt2gt_1,756). happensAt(active(mwt2gt_1),756). orientation(mwt2gt_2,152,756). coords(mwt2gt_2,28,128,756).  visible(mwt2gt_2,756). happensAt(active(mwt2gt_2),756).
orientation(mwt2gt_1,153,757). coords(mwt2gt_1,30,120,757).  visible(mwt2gt_1,757). happensAt(active(mwt2gt_1),757). orientation(mwt2gt_2,152,757). coords(mwt2gt_2,28,128,757).  visible(mwt2gt_2,757). happensAt(active(mwt2gt_2),757).
#show holdsAt/2.