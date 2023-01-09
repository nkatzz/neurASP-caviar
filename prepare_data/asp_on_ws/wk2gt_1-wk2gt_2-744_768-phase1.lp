
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
orientation(wk2gt_1,125,757). coords(wk2gt_1,115,53,757).  visible(wk2gt_1,757). happensAt(inactive(wk2gt_1),757). orientation(wk2gt_2,142,757). coords(wk2gt_2,99,63,757).  visible(wk2gt_2,757). happensAt(inactive(wk2gt_2),757).
orientation(wk2gt_1,125,758). coords(wk2gt_1,115,53,758).  visible(wk2gt_1,758). happensAt(inactive(wk2gt_1),758). orientation(wk2gt_2,142,758). coords(wk2gt_2,99,63,758).  visible(wk2gt_2,758). happensAt(inactive(wk2gt_2),758).
orientation(wk2gt_1,125,759). coords(wk2gt_1,115,53,759).  visible(wk2gt_1,759). happensAt(inactive(wk2gt_1),759). orientation(wk2gt_2,142,759). coords(wk2gt_2,99,63,759).  visible(wk2gt_2,759). happensAt(inactive(wk2gt_2),759).
orientation(wk2gt_1,125,760). coords(wk2gt_1,115,53,760).  visible(wk2gt_1,760). happensAt(inactive(wk2gt_1),760). orientation(wk2gt_2,142,760). coords(wk2gt_2,99,63,760).  visible(wk2gt_2,760). happensAt(inactive(wk2gt_2),760).
orientation(wk2gt_1,125,761). coords(wk2gt_1,115,53,761).  visible(wk2gt_1,761). happensAt(inactive(wk2gt_1),761). orientation(wk2gt_2,142,761). coords(wk2gt_2,99,63,761).  visible(wk2gt_2,761). happensAt(inactive(wk2gt_2),761).
orientation(wk2gt_1,125,762). coords(wk2gt_1,115,53,762).  visible(wk2gt_1,762). happensAt(inactive(wk2gt_1),762). orientation(wk2gt_2,142,762). coords(wk2gt_2,99,63,762).  visible(wk2gt_2,762). happensAt(inactive(wk2gt_2),762).
orientation(wk2gt_1,125,763). coords(wk2gt_1,115,53,763).  visible(wk2gt_1,763). happensAt(inactive(wk2gt_1),763). orientation(wk2gt_2,142,763). coords(wk2gt_2,99,63,763).  visible(wk2gt_2,763). happensAt(inactive(wk2gt_2),763).
orientation(wk2gt_1,125,764). coords(wk2gt_1,115,53,764).  visible(wk2gt_1,764). happensAt(inactive(wk2gt_1),764). orientation(wk2gt_2,142,764). coords(wk2gt_2,99,63,764).  visible(wk2gt_2,764). happensAt(inactive(wk2gt_2),764).
orientation(wk2gt_1,125,765). coords(wk2gt_1,115,53,765).  visible(wk2gt_1,765). happensAt(inactive(wk2gt_1),765). orientation(wk2gt_2,142,765). coords(wk2gt_2,99,63,765).  visible(wk2gt_2,765). happensAt(inactive(wk2gt_2),765).
orientation(wk2gt_1,125,766). coords(wk2gt_1,115,53,766).  visible(wk2gt_1,766). happensAt(inactive(wk2gt_1),766). orientation(wk2gt_2,142,766). coords(wk2gt_2,99,63,766).  visible(wk2gt_2,766). happensAt(inactive(wk2gt_2),766).
orientation(wk2gt_1,125,767). coords(wk2gt_1,115,53,767).  visible(wk2gt_1,767). happensAt(inactive(wk2gt_1),767). orientation(wk2gt_2,142,767). coords(wk2gt_2,99,63,767).  visible(wk2gt_2,767). happensAt(inactive(wk2gt_2),767).
orientation(wk2gt_1,125,768). coords(wk2gt_1,115,53,768).  visible(wk2gt_1,768). happensAt(inactive(wk2gt_1),768). orientation(wk2gt_2,142,768). coords(wk2gt_2,99,63,768).  visible(wk2gt_2,768). happensAt(inactive(wk2gt_2),768).
#show holdsAt/2.