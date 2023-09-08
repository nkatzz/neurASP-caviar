
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
orientation(lbpugt_2,97,756). coords(lbpugt_2,295,99,756).  visible(lbpugt_2,756). happensAt(active(lbpugt_2),756). orientation(lbpugt_4,135,756). coords(lbpugt_4,219,34,756).  visible(lbpugt_4,756). happensAt(inactive(lbpugt_4),756).
orientation(lbpugt_2,97,757). coords(lbpugt_2,295,99,757).  visible(lbpugt_2,757). happensAt(active(lbpugt_2),757). orientation(lbpugt_4,135,757). coords(lbpugt_4,219,34,757).  visible(lbpugt_4,757). happensAt(inactive(lbpugt_4),757).
orientation(lbpugt_2,97,758). coords(lbpugt_2,295,99,758).  visible(lbpugt_2,758). happensAt(active(lbpugt_2),758). orientation(lbpugt_4,135,758). coords(lbpugt_4,219,34,758).  visible(lbpugt_4,758). happensAt(inactive(lbpugt_4),758).
orientation(lbpugt_2,97,759). coords(lbpugt_2,295,99,759).  visible(lbpugt_2,759). happensAt(active(lbpugt_2),759). orientation(lbpugt_4,135,759). coords(lbpugt_4,219,34,759).  visible(lbpugt_4,759). happensAt(inactive(lbpugt_4),759).
orientation(lbpugt_2,97,760). coords(lbpugt_2,295,99,760).  visible(lbpugt_2,760). happensAt(active(lbpugt_2),760). orientation(lbpugt_4,135,760). coords(lbpugt_4,219,34,760).  visible(lbpugt_4,760). happensAt(inactive(lbpugt_4),760).
orientation(lbpugt_2,97,761). coords(lbpugt_2,295,99,761).  visible(lbpugt_2,761). happensAt(active(lbpugt_2),761). orientation(lbpugt_4,135,761). coords(lbpugt_4,219,34,761).  visible(lbpugt_4,761). happensAt(inactive(lbpugt_4),761).
orientation(lbpugt_2,97,762). coords(lbpugt_2,295,99,762).  visible(lbpugt_2,762). happensAt(active(lbpugt_2),762). orientation(lbpugt_4,135,762). coords(lbpugt_4,219,34,762).  visible(lbpugt_4,762). happensAt(inactive(lbpugt_4),762).
orientation(lbpugt_2,97,763). coords(lbpugt_2,295,99,763).  visible(lbpugt_2,763). happensAt(active(lbpugt_2),763). orientation(lbpugt_4,135,763). coords(lbpugt_4,219,34,763).  visible(lbpugt_4,763). happensAt(inactive(lbpugt_4),763).
orientation(lbpugt_2,97,764). coords(lbpugt_2,295,99,764).  visible(lbpugt_2,764). happensAt(active(lbpugt_2),764). orientation(lbpugt_4,135,764). coords(lbpugt_4,219,34,764).  visible(lbpugt_4,764). happensAt(inactive(lbpugt_4),764).
orientation(lbpugt_2,97,765). coords(lbpugt_2,295,99,765).  visible(lbpugt_2,765). happensAt(active(lbpugt_2),765). orientation(lbpugt_4,135,765). coords(lbpugt_4,219,34,765).  visible(lbpugt_4,765). happensAt(inactive(lbpugt_4),765).
orientation(lbpugt_2,97,766). coords(lbpugt_2,295,99,766).  visible(lbpugt_2,766). happensAt(active(lbpugt_2),766). orientation(lbpugt_4,135,766). coords(lbpugt_4,219,34,766).  visible(lbpugt_4,766). happensAt(inactive(lbpugt_4),766).
orientation(lbpugt_2,97,767). coords(lbpugt_2,295,99,767).  visible(lbpugt_2,767). happensAt(active(lbpugt_2),767). orientation(lbpugt_4,135,767). coords(lbpugt_4,219,34,767).  visible(lbpugt_4,767). happensAt(inactive(lbpugt_4),767).
#show holdsAt/2.