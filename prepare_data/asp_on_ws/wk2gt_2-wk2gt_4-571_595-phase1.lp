
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

	orientation(wk2gt_2,142,571). coords(wk2gt_2,99,63,571).  visible(wk2gt_2,571). happensAt(inactive(wk2gt_2),571). orientation(wk2gt_4,147,571). coords(wk2gt_4,42,105,571).  visible(wk2gt_4,571). happensAt(walking(wk2gt_4),571).
orientation(wk2gt_2,142,572). coords(wk2gt_2,99,63,572).  visible(wk2gt_2,572). happensAt(inactive(wk2gt_2),572). orientation(wk2gt_4,147,572). coords(wk2gt_4,42,105,572).  visible(wk2gt_4,572). happensAt(walking(wk2gt_4),572).
orientation(wk2gt_2,142,573). coords(wk2gt_2,99,63,573).  visible(wk2gt_2,573). happensAt(inactive(wk2gt_2),573). orientation(wk2gt_4,147,573). coords(wk2gt_4,42,105,573).  visible(wk2gt_4,573). happensAt(walking(wk2gt_4),573).
orientation(wk2gt_2,142,574). coords(wk2gt_2,99,63,574).  visible(wk2gt_2,574). happensAt(inactive(wk2gt_2),574). orientation(wk2gt_4,147,574). coords(wk2gt_4,42,105,574).  visible(wk2gt_4,574). happensAt(walking(wk2gt_4),574).
orientation(wk2gt_2,142,575). coords(wk2gt_2,99,63,575).  visible(wk2gt_2,575). happensAt(inactive(wk2gt_2),575). orientation(wk2gt_4,147,575). coords(wk2gt_4,42,107,575).  visible(wk2gt_4,575). happensAt(walking(wk2gt_4),575).
orientation(wk2gt_2,142,576). coords(wk2gt_2,99,63,576).  visible(wk2gt_2,576). happensAt(inactive(wk2gt_2),576). orientation(wk2gt_4,152,576). coords(wk2gt_4,42,107,576).  visible(wk2gt_4,576). happensAt(walking(wk2gt_4),576).
orientation(wk2gt_2,142,577). coords(wk2gt_2,99,63,577).  visible(wk2gt_2,577). happensAt(inactive(wk2gt_2),577). orientation(wk2gt_4,152,577). coords(wk2gt_4,42,107,577).  visible(wk2gt_4,577). happensAt(walking(wk2gt_4),577).
orientation(wk2gt_2,142,578). coords(wk2gt_2,99,63,578).  visible(wk2gt_2,578). happensAt(inactive(wk2gt_2),578). orientation(wk2gt_4,152,578). coords(wk2gt_4,40,107,578).  visible(wk2gt_4,578). happensAt(walking(wk2gt_4),578).
orientation(wk2gt_2,142,579). coords(wk2gt_2,99,63,579).  visible(wk2gt_2,579). happensAt(inactive(wk2gt_2),579). orientation(wk2gt_4,152,579). coords(wk2gt_4,40,107,579).  visible(wk2gt_4,579). happensAt(walking(wk2gt_4),579).
orientation(wk2gt_2,142,580). coords(wk2gt_2,99,63,580).  visible(wk2gt_2,580). happensAt(inactive(wk2gt_2),580). orientation(wk2gt_4,152,580). coords(wk2gt_4,40,107,580).  visible(wk2gt_4,580). happensAt(walking(wk2gt_4),580).
orientation(wk2gt_2,142,581). coords(wk2gt_2,99,63,581).  visible(wk2gt_2,581). happensAt(inactive(wk2gt_2),581). orientation(wk2gt_4,152,581). coords(wk2gt_4,39,107,581).  visible(wk2gt_4,581). happensAt(walking(wk2gt_4),581).
orientation(wk2gt_2,142,582). coords(wk2gt_2,99,63,582).  visible(wk2gt_2,582). happensAt(inactive(wk2gt_2),582). orientation(wk2gt_4,152,582). coords(wk2gt_4,39,107,582).  visible(wk2gt_4,582). happensAt(walking(wk2gt_4),582).
orientation(wk2gt_2,142,583). coords(wk2gt_2,99,63,583).  visible(wk2gt_2,583). happensAt(inactive(wk2gt_2),583). orientation(wk2gt_4,152,583). coords(wk2gt_4,39,108,583).  visible(wk2gt_4,583). happensAt(walking(wk2gt_4),583).
orientation(wk2gt_2,142,584). coords(wk2gt_2,99,63,584).  visible(wk2gt_2,584). happensAt(inactive(wk2gt_2),584). orientation(wk2gt_4,152,584). coords(wk2gt_4,39,108,584).  visible(wk2gt_4,584). happensAt(walking(wk2gt_4),584).
orientation(wk2gt_2,142,585). coords(wk2gt_2,99,63,585).  visible(wk2gt_2,585). happensAt(inactive(wk2gt_2),585). orientation(wk2gt_4,152,585). coords(wk2gt_4,39,109,585).  visible(wk2gt_4,585). happensAt(walking(wk2gt_4),585).
orientation(wk2gt_2,142,586). coords(wk2gt_2,99,63,586).  visible(wk2gt_2,586). happensAt(inactive(wk2gt_2),586). orientation(wk2gt_4,152,586). coords(wk2gt_4,38,110,586).  visible(wk2gt_4,586). happensAt(walking(wk2gt_4),586).
orientation(wk2gt_2,142,587). coords(wk2gt_2,99,63,587).  visible(wk2gt_2,587). happensAt(inactive(wk2gt_2),587). orientation(wk2gt_4,152,587). coords(wk2gt_4,38,110,587).  visible(wk2gt_4,587). happensAt(walking(wk2gt_4),587).
orientation(wk2gt_2,142,588). coords(wk2gt_2,99,63,588).  visible(wk2gt_2,588). happensAt(inactive(wk2gt_2),588). orientation(wk2gt_4,152,588). coords(wk2gt_4,37,110,588).  visible(wk2gt_4,588). happensAt(walking(wk2gt_4),588).
orientation(wk2gt_2,142,589). coords(wk2gt_2,99,63,589).  visible(wk2gt_2,589). happensAt(inactive(wk2gt_2),589). orientation(wk2gt_4,152,589). coords(wk2gt_4,37,110,589).  visible(wk2gt_4,589). happensAt(walking(wk2gt_4),589).
orientation(wk2gt_2,142,590). coords(wk2gt_2,99,63,590).  visible(wk2gt_2,590). happensAt(inactive(wk2gt_2),590). orientation(wk2gt_4,152,590). coords(wk2gt_4,36,110,590).  visible(wk2gt_4,590). happensAt(walking(wk2gt_4),590).
orientation(wk2gt_2,142,591). coords(wk2gt_2,99,63,591).  visible(wk2gt_2,591). happensAt(inactive(wk2gt_2),591). orientation(wk2gt_4,152,591). coords(wk2gt_4,36,111,591).  visible(wk2gt_4,591). happensAt(walking(wk2gt_4),591).
orientation(wk2gt_2,142,592). coords(wk2gt_2,99,63,592).  visible(wk2gt_2,592). happensAt(inactive(wk2gt_2),592). orientation(wk2gt_4,152,592). coords(wk2gt_4,36,111,592).  visible(wk2gt_4,592). happensAt(walking(wk2gt_4),592).
orientation(wk2gt_2,142,593). coords(wk2gt_2,99,63,593).  visible(wk2gt_2,593). happensAt(inactive(wk2gt_2),593). orientation(wk2gt_4,152,593). coords(wk2gt_4,35,111,593).  visible(wk2gt_4,593). happensAt(walking(wk2gt_4),593).
orientation(wk2gt_2,142,594). coords(wk2gt_2,99,63,594).  visible(wk2gt_2,594). happensAt(inactive(wk2gt_2),594). orientation(wk2gt_4,152,594). coords(wk2gt_4,35,111,594).  visible(wk2gt_4,594). happensAt(walking(wk2gt_4),594).
orientation(wk2gt_2,142,595). coords(wk2gt_2,99,63,595).  visible(wk2gt_2,595). happensAt(inactive(wk2gt_2),595). orientation(wk2gt_4,152,595). coords(wk2gt_4,35,111,595).  visible(wk2gt_4,595). happensAt(walking(wk2gt_4),595).
#show holdsAt/2.