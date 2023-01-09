
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

	orientation(lbpugt_2,102,731). coords(lbpugt_2,293,97,731).  visible(lbpugt_2,731). happensAt(walking(lbpugt_2),731). orientation(lbpugt_4,135,731). coords(lbpugt_4,219,34,731).  visible(lbpugt_4,731). happensAt(inactive(lbpugt_4),731).
orientation(lbpugt_2,99,732). coords(lbpugt_2,294,97,732).  visible(lbpugt_2,732). happensAt(walking(lbpugt_2),732). orientation(lbpugt_4,135,732). coords(lbpugt_4,219,34,732).  visible(lbpugt_4,732). happensAt(inactive(lbpugt_4),732).
orientation(lbpugt_2,101,733). coords(lbpugt_2,293,97,733).  visible(lbpugt_2,733). happensAt(walking(lbpugt_2),733). orientation(lbpugt_4,135,733). coords(lbpugt_4,219,34,733).  visible(lbpugt_4,733). happensAt(inactive(lbpugt_4),733).
orientation(lbpugt_2,97,734). coords(lbpugt_2,294,97,734).  visible(lbpugt_2,734). happensAt(walking(lbpugt_2),734). orientation(lbpugt_4,135,734). coords(lbpugt_4,219,34,734).  visible(lbpugt_4,734). happensAt(inactive(lbpugt_4),734).
orientation(lbpugt_2,97,735). coords(lbpugt_2,294,97,735).  visible(lbpugt_2,735). happensAt(walking(lbpugt_2),735). orientation(lbpugt_4,135,735). coords(lbpugt_4,219,34,735).  visible(lbpugt_4,735). happensAt(inactive(lbpugt_4),735).
orientation(lbpugt_2,97,736). coords(lbpugt_2,294,97,736).  visible(lbpugt_2,736). happensAt(walking(lbpugt_2),736). orientation(lbpugt_4,135,736). coords(lbpugt_4,219,34,736).  visible(lbpugt_4,736). happensAt(inactive(lbpugt_4),736).
orientation(lbpugt_2,97,737). coords(lbpugt_2,294,97,737).  visible(lbpugt_2,737). happensAt(walking(lbpugt_2),737). orientation(lbpugt_4,135,737). coords(lbpugt_4,219,34,737).  visible(lbpugt_4,737). happensAt(inactive(lbpugt_4),737).
orientation(lbpugt_2,97,738). coords(lbpugt_2,294,98,738).  visible(lbpugt_2,738). happensAt(active(lbpugt_2),738). orientation(lbpugt_4,135,738). coords(lbpugt_4,219,34,738).  visible(lbpugt_4,738). happensAt(inactive(lbpugt_4),738).
orientation(lbpugt_2,97,739). coords(lbpugt_2,294,98,739).  visible(lbpugt_2,739). happensAt(active(lbpugt_2),739). orientation(lbpugt_4,135,739). coords(lbpugt_4,219,34,739).  visible(lbpugt_4,739). happensAt(inactive(lbpugt_4),739).
orientation(lbpugt_2,97,740). coords(lbpugt_2,294,98,740).  visible(lbpugt_2,740). happensAt(active(lbpugt_2),740). orientation(lbpugt_4,135,740). coords(lbpugt_4,219,34,740).  visible(lbpugt_4,740). happensAt(inactive(lbpugt_4),740).
orientation(lbpugt_2,97,741). coords(lbpugt_2,294,98,741).  visible(lbpugt_2,741). happensAt(active(lbpugt_2),741). orientation(lbpugt_4,135,741). coords(lbpugt_4,219,34,741).  visible(lbpugt_4,741). happensAt(inactive(lbpugt_4),741).
orientation(lbpugt_2,97,742). coords(lbpugt_2,294,98,742).  visible(lbpugt_2,742). happensAt(active(lbpugt_2),742). orientation(lbpugt_4,135,742). coords(lbpugt_4,219,34,742).  visible(lbpugt_4,742). happensAt(inactive(lbpugt_4),742).
orientation(lbpugt_2,97,743). coords(lbpugt_2,294,98,743).  visible(lbpugt_2,743). happensAt(active(lbpugt_2),743). orientation(lbpugt_4,135,743). coords(lbpugt_4,219,34,743).  visible(lbpugt_4,743). happensAt(inactive(lbpugt_4),743).
orientation(lbpugt_2,97,744). coords(lbpugt_2,294,98,744).  visible(lbpugt_2,744). happensAt(active(lbpugt_2),744). orientation(lbpugt_4,135,744). coords(lbpugt_4,219,34,744).  visible(lbpugt_4,744). happensAt(inactive(lbpugt_4),744).
orientation(lbpugt_2,97,745). coords(lbpugt_2,294,98,745).  visible(lbpugt_2,745). happensAt(active(lbpugt_2),745). orientation(lbpugt_4,135,745). coords(lbpugt_4,219,34,745).  visible(lbpugt_4,745). happensAt(inactive(lbpugt_4),745).
orientation(lbpugt_2,97,746). coords(lbpugt_2,294,98,746).  visible(lbpugt_2,746). happensAt(active(lbpugt_2),746). orientation(lbpugt_4,135,746). coords(lbpugt_4,219,34,746).  visible(lbpugt_4,746). happensAt(inactive(lbpugt_4),746).
orientation(lbpugt_2,97,747). coords(lbpugt_2,294,98,747).  visible(lbpugt_2,747). happensAt(active(lbpugt_2),747). orientation(lbpugt_4,135,747). coords(lbpugt_4,219,34,747).  visible(lbpugt_4,747). happensAt(inactive(lbpugt_4),747).
orientation(lbpugt_2,97,748). coords(lbpugt_2,294,98,748).  visible(lbpugt_2,748). happensAt(active(lbpugt_2),748). orientation(lbpugt_4,135,748). coords(lbpugt_4,219,34,748).  visible(lbpugt_4,748). happensAt(inactive(lbpugt_4),748).
orientation(lbpugt_2,97,749). coords(lbpugt_2,294,98,749).  visible(lbpugt_2,749). happensAt(active(lbpugt_2),749). orientation(lbpugt_4,135,749). coords(lbpugt_4,219,34,749).  visible(lbpugt_4,749). happensAt(inactive(lbpugt_4),749).
orientation(lbpugt_2,97,750). coords(lbpugt_2,294,98,750).  visible(lbpugt_2,750). happensAt(active(lbpugt_2),750). orientation(lbpugt_4,135,750). coords(lbpugt_4,219,34,750).  visible(lbpugt_4,750). happensAt(inactive(lbpugt_4),750).
orientation(lbpugt_2,97,751). coords(lbpugt_2,294,98,751).  visible(lbpugt_2,751). happensAt(active(lbpugt_2),751). orientation(lbpugt_4,135,751). coords(lbpugt_4,219,34,751).  visible(lbpugt_4,751). happensAt(inactive(lbpugt_4),751).
orientation(lbpugt_2,97,752). coords(lbpugt_2,294,98,752).  visible(lbpugt_2,752). happensAt(active(lbpugt_2),752). orientation(lbpugt_4,135,752). coords(lbpugt_4,219,34,752).  visible(lbpugt_4,752). happensAt(inactive(lbpugt_4),752).
orientation(lbpugt_2,97,753). coords(lbpugt_2,295,98,753).  visible(lbpugt_2,753). happensAt(active(lbpugt_2),753). orientation(lbpugt_4,135,753). coords(lbpugt_4,219,34,753).  visible(lbpugt_4,753). happensAt(inactive(lbpugt_4),753).
orientation(lbpugt_2,97,754). coords(lbpugt_2,295,98,754).  visible(lbpugt_2,754). happensAt(active(lbpugt_2),754). orientation(lbpugt_4,135,754). coords(lbpugt_4,219,34,754).  visible(lbpugt_4,754). happensAt(inactive(lbpugt_4),754).
orientation(lbpugt_2,97,755). coords(lbpugt_2,295,98,755).  visible(lbpugt_2,755). happensAt(active(lbpugt_2),755). orientation(lbpugt_4,135,755). coords(lbpugt_4,219,34,755).  visible(lbpugt_4,755). happensAt(inactive(lbpugt_4),755).
#show holdsAt/2.