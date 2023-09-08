
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
orientation(br1gt_2,12,587). coords(br1gt_2,86,209,587).  visible(br1gt_2,587). happensAt(inactive(br1gt_2),587). orientation(br1gt_3,134,587). coords(br1gt_3,254,99,587).  visible(br1gt_3,587). happensAt(walking(br1gt_3),587).
orientation(br1gt_2,12,588). coords(br1gt_2,86,209,588).  visible(br1gt_2,588). happensAt(inactive(br1gt_2),588). orientation(br1gt_3,134,588). coords(br1gt_3,255,98,588).  visible(br1gt_3,588). happensAt(walking(br1gt_3),588).
orientation(br1gt_2,12,589). coords(br1gt_2,86,209,589).  visible(br1gt_2,589). happensAt(inactive(br1gt_2),589). orientation(br1gt_3,134,589). coords(br1gt_3,256,99,589).  visible(br1gt_3,589). happensAt(walking(br1gt_3),589).
orientation(br1gt_2,12,590). coords(br1gt_2,86,209,590).  visible(br1gt_2,590). happensAt(inactive(br1gt_2),590). orientation(br1gt_3,134,590). coords(br1gt_3,257,98,590).  visible(br1gt_3,590). happensAt(walking(br1gt_3),590).
orientation(br1gt_2,12,591). coords(br1gt_2,86,209,591).  visible(br1gt_2,591). happensAt(inactive(br1gt_2),591). orientation(br1gt_3,126,591). coords(br1gt_3,260,98,591).  visible(br1gt_3,591). happensAt(walking(br1gt_3),591).
orientation(br1gt_2,12,592). coords(br1gt_2,86,209,592).  visible(br1gt_2,592). happensAt(inactive(br1gt_2),592). orientation(br1gt_3,126,592). coords(br1gt_3,264,98,592).  visible(br1gt_3,592). happensAt(walking(br1gt_3),592).
orientation(br1gt_2,12,593). coords(br1gt_2,86,209,593).  visible(br1gt_2,593). happensAt(inactive(br1gt_2),593). orientation(br1gt_3,129,593). coords(br1gt_3,267,98,593).  visible(br1gt_3,593). happensAt(walking(br1gt_3),593).
orientation(br1gt_2,12,594). coords(br1gt_2,86,209,594).  visible(br1gt_2,594). happensAt(inactive(br1gt_2),594). orientation(br1gt_3,129,594). coords(br1gt_3,270,98,594).  visible(br1gt_3,594). happensAt(walking(br1gt_3),594).
orientation(br1gt_2,12,595). coords(br1gt_2,86,209,595).  visible(br1gt_2,595). happensAt(inactive(br1gt_2),595). orientation(br1gt_3,129,595). coords(br1gt_3,272,98,595).  visible(br1gt_3,595). happensAt(walking(br1gt_3),595).
orientation(br1gt_2,12,596). coords(br1gt_2,86,209,596).  visible(br1gt_2,596). happensAt(inactive(br1gt_2),596). orientation(br1gt_3,129,596). coords(br1gt_3,274,97,596).  visible(br1gt_3,596). happensAt(walking(br1gt_3),596).
orientation(br1gt_2,12,597). coords(br1gt_2,86,209,597).  visible(br1gt_2,597). happensAt(inactive(br1gt_2),597). orientation(br1gt_3,129,597). coords(br1gt_3,275,97,597).  visible(br1gt_3,597). happensAt(walking(br1gt_3),597).
orientation(br1gt_2,12,598). coords(br1gt_2,86,209,598).  visible(br1gt_2,598). happensAt(inactive(br1gt_2),598). orientation(br1gt_3,129,598). coords(br1gt_3,276,97,598).  visible(br1gt_3,598). happensAt(walking(br1gt_3),598).
#show holdsAt/2.