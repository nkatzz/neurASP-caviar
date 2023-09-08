
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

	orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280).
orientation(fcgt_1,153,281). coords(fcgt_1,119,129,281).  visible(fcgt_1,281). happensAt(walking(fcgt_1),281). orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281).
orientation(fcgt_1,153,282). coords(fcgt_1,118,127,282).  visible(fcgt_1,282). happensAt(walking(fcgt_1),282). orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282).
orientation(fcgt_1,153,283). coords(fcgt_1,115,126,283).  visible(fcgt_1,283). happensAt(walking(fcgt_1),283). orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283).
orientation(fcgt_1,153,284). coords(fcgt_1,113,126,284).  visible(fcgt_1,284). happensAt(walking(fcgt_1),284). orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284).
orientation(fcgt_1,153,285). coords(fcgt_1,112,127,285).  visible(fcgt_1,285). happensAt(walking(fcgt_1),285). orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285).
orientation(fcgt_1,153,286). coords(fcgt_1,111,127,286).  visible(fcgt_1,286). happensAt(walking(fcgt_1),286). orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286).
orientation(fcgt_1,153,287). coords(fcgt_1,111,127,287).  visible(fcgt_1,287). happensAt(walking(fcgt_1),287). orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287).
orientation(fcgt_1,153,288). coords(fcgt_1,110,127,288).  visible(fcgt_1,288). happensAt(walking(fcgt_1),288). orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288).
orientation(fcgt_1,153,289). coords(fcgt_1,110,127,289).  visible(fcgt_1,289). happensAt(walking(fcgt_1),289). orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289).
orientation(fcgt_1,153,290). coords(fcgt_1,110,128,290).  visible(fcgt_1,290). happensAt(walking(fcgt_1),290). orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290).
orientation(fcgt_1,153,291). coords(fcgt_1,109,128,291).  visible(fcgt_1,291). happensAt(walking(fcgt_1),291). orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291).
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). happensAt(walking(fcgt_1),292). orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292).
orientation(fcgt_1,153,293). coords(fcgt_1,107,126,293).  visible(fcgt_1,293). happensAt(walking(fcgt_1),293). orientation(fcgt_3,164,293). coords(fcgt_3,98,186,293).  visible(fcgt_3,293). happensAt(inactive(fcgt_3),293).
orientation(fcgt_1,153,294). coords(fcgt_1,104,126,294).  visible(fcgt_1,294). happensAt(walking(fcgt_1),294). orientation(fcgt_3,164,294). coords(fcgt_3,98,186,294).  visible(fcgt_3,294). happensAt(inactive(fcgt_3),294).
orientation(fcgt_1,153,295). coords(fcgt_1,103,127,295).  visible(fcgt_1,295). happensAt(walking(fcgt_1),295). orientation(fcgt_3,164,295). coords(fcgt_3,98,186,295).  visible(fcgt_3,295). happensAt(inactive(fcgt_3),295).
orientation(fcgt_1,153,296). coords(fcgt_1,101,125,296).  visible(fcgt_1,296). happensAt(walking(fcgt_1),296). orientation(fcgt_3,164,296). coords(fcgt_3,98,186,296).  visible(fcgt_3,296). happensAt(inactive(fcgt_3),296).
orientation(fcgt_1,153,297). coords(fcgt_1,101,124,297).  visible(fcgt_1,297). happensAt(active(fcgt_1),297). orientation(fcgt_3,164,297). coords(fcgt_3,98,187,297).  visible(fcgt_3,297). happensAt(inactive(fcgt_3),297).
orientation(fcgt_1,153,298). coords(fcgt_1,101,123,298).  visible(fcgt_1,298). happensAt(active(fcgt_1),298). orientation(fcgt_3,164,298). coords(fcgt_3,98,187,298).  visible(fcgt_3,298). happensAt(inactive(fcgt_3),298).
orientation(fcgt_1,153,299). coords(fcgt_1,100,123,299).  visible(fcgt_1,299). happensAt(active(fcgt_1),299). orientation(fcgt_3,164,299). coords(fcgt_3,98,187,299).  visible(fcgt_3,299). happensAt(inactive(fcgt_3),299).
orientation(fcgt_1,153,300). coords(fcgt_1,100,123,300).  visible(fcgt_1,300). happensAt(active(fcgt_1),300). orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). happensAt(inactive(fcgt_3),300).
orientation(fcgt_1,153,301). coords(fcgt_1,99,123,301).  visible(fcgt_1,301). happensAt(active(fcgt_1),301). orientation(fcgt_3,164,301). coords(fcgt_3,98,187,301).  visible(fcgt_3,301). happensAt(inactive(fcgt_3),301).
orientation(fcgt_1,153,302). coords(fcgt_1,99,124,302).  visible(fcgt_1,302). happensAt(active(fcgt_1),302). orientation(fcgt_3,164,302). coords(fcgt_3,98,187,302).  visible(fcgt_3,302). happensAt(inactive(fcgt_3),302).
orientation(fcgt_1,153,303). coords(fcgt_1,98,125,303).  visible(fcgt_1,303). happensAt(active(fcgt_1),303). orientation(fcgt_3,164,303). coords(fcgt_3,98,187,303).  visible(fcgt_3,303). happensAt(inactive(fcgt_3),303).
orientation(fcgt_1,153,304). coords(fcgt_1,98,127,304).  visible(fcgt_1,304). happensAt(active(fcgt_1),304). orientation(fcgt_3,164,304). coords(fcgt_3,98,187,304).  visible(fcgt_3,304). happensAt(inactive(fcgt_3),304).
#show holdsAt/2.