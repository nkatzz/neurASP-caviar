
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

	orientation(wk2gt_1,125,564). coords(wk2gt_1,115,53,564).  visible(wk2gt_1,564). happensAt(inactive(wk2gt_1),564). orientation(wk2gt_2,142,564). coords(wk2gt_2,99,63,564).  visible(wk2gt_2,564). happensAt(inactive(wk2gt_2),564).
orientation(wk2gt_1,125,565). coords(wk2gt_1,115,53,565).  visible(wk2gt_1,565). happensAt(inactive(wk2gt_1),565). orientation(wk2gt_2,142,565). coords(wk2gt_2,99,63,565).  visible(wk2gt_2,565). happensAt(inactive(wk2gt_2),565).
orientation(wk2gt_1,125,566). coords(wk2gt_1,115,53,566).  visible(wk2gt_1,566). happensAt(inactive(wk2gt_1),566). orientation(wk2gt_2,142,566). coords(wk2gt_2,99,63,566).  visible(wk2gt_2,566). happensAt(inactive(wk2gt_2),566).
orientation(wk2gt_1,125,567). coords(wk2gt_1,115,53,567).  visible(wk2gt_1,567). happensAt(inactive(wk2gt_1),567). orientation(wk2gt_2,142,567). coords(wk2gt_2,99,63,567).  visible(wk2gt_2,567). happensAt(inactive(wk2gt_2),567).
orientation(wk2gt_1,125,568). coords(wk2gt_1,115,53,568).  visible(wk2gt_1,568). happensAt(inactive(wk2gt_1),568). orientation(wk2gt_2,142,568). coords(wk2gt_2,99,63,568).  visible(wk2gt_2,568). happensAt(inactive(wk2gt_2),568).
orientation(wk2gt_1,125,569). coords(wk2gt_1,115,53,569).  visible(wk2gt_1,569). happensAt(inactive(wk2gt_1),569). orientation(wk2gt_2,142,569). coords(wk2gt_2,99,63,569).  visible(wk2gt_2,569). happensAt(inactive(wk2gt_2),569).
orientation(wk2gt_1,125,570). coords(wk2gt_1,115,53,570).  visible(wk2gt_1,570). happensAt(inactive(wk2gt_1),570). orientation(wk2gt_2,142,570). coords(wk2gt_2,99,63,570).  visible(wk2gt_2,570). happensAt(inactive(wk2gt_2),570).
orientation(wk2gt_1,125,571). coords(wk2gt_1,115,53,571).  visible(wk2gt_1,571). happensAt(inactive(wk2gt_1),571). orientation(wk2gt_2,142,571). coords(wk2gt_2,99,63,571).  visible(wk2gt_2,571). happensAt(inactive(wk2gt_2),571).
orientation(wk2gt_1,125,572). coords(wk2gt_1,115,53,572).  visible(wk2gt_1,572). happensAt(inactive(wk2gt_1),572). orientation(wk2gt_2,142,572). coords(wk2gt_2,99,63,572).  visible(wk2gt_2,572). happensAt(inactive(wk2gt_2),572).
orientation(wk2gt_1,125,573). coords(wk2gt_1,115,53,573).  visible(wk2gt_1,573). happensAt(inactive(wk2gt_1),573). orientation(wk2gt_2,142,573). coords(wk2gt_2,99,63,573).  visible(wk2gt_2,573). happensAt(inactive(wk2gt_2),573).
orientation(wk2gt_1,125,574). coords(wk2gt_1,115,53,574).  visible(wk2gt_1,574). happensAt(inactive(wk2gt_1),574). orientation(wk2gt_2,142,574). coords(wk2gt_2,99,63,574).  visible(wk2gt_2,574). happensAt(inactive(wk2gt_2),574).
orientation(wk2gt_1,125,575). coords(wk2gt_1,115,53,575).  visible(wk2gt_1,575). happensAt(inactive(wk2gt_1),575). orientation(wk2gt_2,142,575). coords(wk2gt_2,99,63,575).  visible(wk2gt_2,575). happensAt(inactive(wk2gt_2),575).
orientation(wk2gt_1,125,576). coords(wk2gt_1,115,53,576).  visible(wk2gt_1,576). happensAt(inactive(wk2gt_1),576). orientation(wk2gt_2,142,576). coords(wk2gt_2,99,63,576).  visible(wk2gt_2,576). happensAt(inactive(wk2gt_2),576).
orientation(wk2gt_1,125,577). coords(wk2gt_1,115,53,577).  visible(wk2gt_1,577). happensAt(inactive(wk2gt_1),577). orientation(wk2gt_2,142,577). coords(wk2gt_2,99,63,577).  visible(wk2gt_2,577). happensAt(inactive(wk2gt_2),577).
orientation(wk2gt_1,125,578). coords(wk2gt_1,115,53,578).  visible(wk2gt_1,578). happensAt(inactive(wk2gt_1),578). orientation(wk2gt_2,142,578). coords(wk2gt_2,99,63,578).  visible(wk2gt_2,578). happensAt(inactive(wk2gt_2),578).
orientation(wk2gt_1,125,579). coords(wk2gt_1,115,53,579).  visible(wk2gt_1,579). happensAt(inactive(wk2gt_1),579). orientation(wk2gt_2,142,579). coords(wk2gt_2,99,63,579).  visible(wk2gt_2,579). happensAt(inactive(wk2gt_2),579).
orientation(wk2gt_1,125,580). coords(wk2gt_1,115,53,580).  visible(wk2gt_1,580). happensAt(inactive(wk2gt_1),580). orientation(wk2gt_2,142,580). coords(wk2gt_2,99,63,580).  visible(wk2gt_2,580). happensAt(inactive(wk2gt_2),580).
orientation(wk2gt_1,125,581). coords(wk2gt_1,115,53,581).  visible(wk2gt_1,581). happensAt(inactive(wk2gt_1),581). orientation(wk2gt_2,142,581). coords(wk2gt_2,99,63,581).  visible(wk2gt_2,581). happensAt(inactive(wk2gt_2),581).
orientation(wk2gt_1,125,582). coords(wk2gt_1,115,53,582).  visible(wk2gt_1,582). happensAt(inactive(wk2gt_1),582). orientation(wk2gt_2,142,582). coords(wk2gt_2,99,63,582).  visible(wk2gt_2,582). happensAt(inactive(wk2gt_2),582).
orientation(wk2gt_1,125,583). coords(wk2gt_1,115,53,583).  visible(wk2gt_1,583). happensAt(inactive(wk2gt_1),583). orientation(wk2gt_2,142,583). coords(wk2gt_2,99,63,583).  visible(wk2gt_2,583). happensAt(inactive(wk2gt_2),583).
orientation(wk2gt_1,125,584). coords(wk2gt_1,115,53,584).  visible(wk2gt_1,584). happensAt(inactive(wk2gt_1),584). orientation(wk2gt_2,142,584). coords(wk2gt_2,99,63,584).  visible(wk2gt_2,584). happensAt(inactive(wk2gt_2),584).
orientation(wk2gt_1,125,585). coords(wk2gt_1,115,53,585).  visible(wk2gt_1,585). happensAt(inactive(wk2gt_1),585). orientation(wk2gt_2,142,585). coords(wk2gt_2,99,63,585).  visible(wk2gt_2,585). happensAt(inactive(wk2gt_2),585).
orientation(wk2gt_1,125,586). coords(wk2gt_1,115,53,586).  visible(wk2gt_1,586). happensAt(inactive(wk2gt_1),586). orientation(wk2gt_2,142,586). coords(wk2gt_2,99,63,586).  visible(wk2gt_2,586). happensAt(inactive(wk2gt_2),586).
orientation(wk2gt_1,125,587). coords(wk2gt_1,115,53,587).  visible(wk2gt_1,587). happensAt(inactive(wk2gt_1),587). orientation(wk2gt_2,142,587). coords(wk2gt_2,99,63,587).  visible(wk2gt_2,587). happensAt(inactive(wk2gt_2),587).
orientation(wk2gt_1,125,588). coords(wk2gt_1,115,53,588).  visible(wk2gt_1,588). happensAt(inactive(wk2gt_1),588). orientation(wk2gt_2,142,588). coords(wk2gt_2,99,63,588).  visible(wk2gt_2,588). happensAt(inactive(wk2gt_2),588).
#show holdsAt/2.