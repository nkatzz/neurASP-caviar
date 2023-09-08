
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

	orientation(br1gt_2,12,562). coords(br1gt_2,86,209,562).  visible(br1gt_2,562). happensAt(inactive(br1gt_2),562). orientation(br1gt_3,145,562). coords(br1gt_3,202,108,562).  visible(br1gt_3,562). happensAt(walking(br1gt_3),562).
orientation(br1gt_2,12,563). coords(br1gt_2,86,209,563).  visible(br1gt_2,563). happensAt(inactive(br1gt_2),563). orientation(br1gt_3,145,563). coords(br1gt_3,203,108,563).  visible(br1gt_3,563). happensAt(walking(br1gt_3),563).
orientation(br1gt_2,12,564). coords(br1gt_2,86,209,564).  visible(br1gt_2,564). happensAt(inactive(br1gt_2),564). orientation(br1gt_3,145,564). coords(br1gt_3,207,108,564).  visible(br1gt_3,564). happensAt(walking(br1gt_3),564).
orientation(br1gt_2,12,565). coords(br1gt_2,86,209,565).  visible(br1gt_2,565). happensAt(inactive(br1gt_2),565). orientation(br1gt_3,145,565). coords(br1gt_3,211,107,565).  visible(br1gt_3,565). happensAt(walking(br1gt_3),565).
orientation(br1gt_2,12,566). coords(br1gt_2,86,209,566).  visible(br1gt_2,566). happensAt(inactive(br1gt_2),566). orientation(br1gt_3,145,566). coords(br1gt_3,214,107,566).  visible(br1gt_3,566). happensAt(walking(br1gt_3),566).
orientation(br1gt_2,12,567). coords(br1gt_2,86,209,567).  visible(br1gt_2,567). happensAt(inactive(br1gt_2),567). orientation(br1gt_3,145,567). coords(br1gt_3,217,107,567).  visible(br1gt_3,567). happensAt(walking(br1gt_3),567).
orientation(br1gt_2,12,568). coords(br1gt_2,86,209,568).  visible(br1gt_2,568). happensAt(inactive(br1gt_2),568). orientation(br1gt_3,145,568). coords(br1gt_3,219,107,568).  visible(br1gt_3,568). happensAt(walking(br1gt_3),568).
orientation(br1gt_2,12,569). coords(br1gt_2,86,209,569).  visible(br1gt_2,569). happensAt(inactive(br1gt_2),569). orientation(br1gt_3,145,569). coords(br1gt_3,222,106,569).  visible(br1gt_3,569). happensAt(walking(br1gt_3),569).
orientation(br1gt_2,12,570). coords(br1gt_2,86,209,570).  visible(br1gt_2,570). happensAt(inactive(br1gt_2),570). orientation(br1gt_3,145,570). coords(br1gt_3,224,106,570).  visible(br1gt_3,570). happensAt(walking(br1gt_3),570).
orientation(br1gt_2,12,571). coords(br1gt_2,86,209,571).  visible(br1gt_2,571). happensAt(inactive(br1gt_2),571). orientation(br1gt_3,145,571). coords(br1gt_3,224,105,571).  visible(br1gt_3,571). happensAt(walking(br1gt_3),571).
orientation(br1gt_2,12,572). coords(br1gt_2,86,209,572).  visible(br1gt_2,572). happensAt(inactive(br1gt_2),572). orientation(br1gt_3,138,572). coords(br1gt_3,226,105,572).  visible(br1gt_3,572). happensAt(walking(br1gt_3),572).
orientation(br1gt_2,12,573). coords(br1gt_2,86,209,573).  visible(br1gt_2,573). happensAt(inactive(br1gt_2),573). orientation(br1gt_3,138,573). coords(br1gt_3,227,104,573).  visible(br1gt_3,573). happensAt(walking(br1gt_3),573).
orientation(br1gt_2,12,574). coords(br1gt_2,86,209,574).  visible(br1gt_2,574). happensAt(inactive(br1gt_2),574). orientation(br1gt_3,138,574). coords(br1gt_3,228,103,574).  visible(br1gt_3,574). happensAt(walking(br1gt_3),574).
orientation(br1gt_2,12,575). coords(br1gt_2,86,209,575).  visible(br1gt_2,575). happensAt(inactive(br1gt_2),575). orientation(br1gt_3,133,575). coords(br1gt_3,229,102,575).  visible(br1gt_3,575). happensAt(walking(br1gt_3),575).
orientation(br1gt_2,12,576). coords(br1gt_2,86,209,576).  visible(br1gt_2,576). happensAt(inactive(br1gt_2),576). orientation(br1gt_3,133,576). coords(br1gt_3,230,102,576).  visible(br1gt_3,576). happensAt(walking(br1gt_3),576).
orientation(br1gt_2,12,577). coords(br1gt_2,86,209,577).  visible(br1gt_2,577). happensAt(inactive(br1gt_2),577). orientation(br1gt_3,131,577). coords(br1gt_3,232,101,577).  visible(br1gt_3,577). happensAt(walking(br1gt_3),577).
orientation(br1gt_2,12,578). coords(br1gt_2,86,209,578).  visible(br1gt_2,578). happensAt(inactive(br1gt_2),578). orientation(br1gt_3,131,578). coords(br1gt_3,236,101,578).  visible(br1gt_3,578). happensAt(walking(br1gt_3),578).
orientation(br1gt_2,12,579). coords(br1gt_2,86,209,579).  visible(br1gt_2,579). happensAt(inactive(br1gt_2),579). orientation(br1gt_3,131,579). coords(br1gt_3,240,101,579).  visible(br1gt_3,579). happensAt(walking(br1gt_3),579).
orientation(br1gt_2,12,580). coords(br1gt_2,86,209,580).  visible(br1gt_2,580). happensAt(inactive(br1gt_2),580). orientation(br1gt_3,136,580). coords(br1gt_3,243,100,580).  visible(br1gt_3,580). happensAt(walking(br1gt_3),580).
orientation(br1gt_2,12,581). coords(br1gt_2,86,209,581).  visible(br1gt_2,581). happensAt(inactive(br1gt_2),581). orientation(br1gt_3,136,581). coords(br1gt_3,246,99,581).  visible(br1gt_3,581). happensAt(walking(br1gt_3),581).
orientation(br1gt_2,12,582). coords(br1gt_2,86,209,582).  visible(br1gt_2,582). happensAt(inactive(br1gt_2),582). orientation(br1gt_3,139,582). coords(br1gt_3,248,99,582).  visible(br1gt_3,582). happensAt(walking(br1gt_3),582).
orientation(br1gt_2,12,583). coords(br1gt_2,86,209,583).  visible(br1gt_2,583). happensAt(inactive(br1gt_2),583). orientation(br1gt_3,139,583). coords(br1gt_3,249,99,583).  visible(br1gt_3,583). happensAt(walking(br1gt_3),583).
orientation(br1gt_2,12,584). coords(br1gt_2,86,209,584).  visible(br1gt_2,584). happensAt(inactive(br1gt_2),584). orientation(br1gt_3,139,584). coords(br1gt_3,251,99,584).  visible(br1gt_3,584). happensAt(walking(br1gt_3),584).
orientation(br1gt_2,12,585). coords(br1gt_2,86,209,585).  visible(br1gt_2,585). happensAt(inactive(br1gt_2),585). orientation(br1gt_3,139,585). coords(br1gt_3,252,99,585).  visible(br1gt_3,585). happensAt(walking(br1gt_3),585).
orientation(br1gt_2,12,586). coords(br1gt_2,86,209,586).  visible(br1gt_2,586). happensAt(inactive(br1gt_2),586). orientation(br1gt_3,139,586). coords(br1gt_3,253,99,586).  visible(br1gt_3,586). happensAt(walking(br1gt_3),586).
#show holdsAt/2.