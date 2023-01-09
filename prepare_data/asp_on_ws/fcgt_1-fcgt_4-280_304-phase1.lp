
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

	orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
orientation(fcgt_1,153,281). coords(fcgt_1,119,129,281).  visible(fcgt_1,281). happensAt(walking(fcgt_1),281). orientation(fcgt_4,166,281). coords(fcgt_4,81,205,281).  visible(fcgt_4,281). happensAt(inactive(fcgt_4),281).
orientation(fcgt_1,153,282). coords(fcgt_1,118,127,282).  visible(fcgt_1,282). happensAt(walking(fcgt_1),282). orientation(fcgt_4,166,282). coords(fcgt_4,81,205,282).  visible(fcgt_4,282). happensAt(inactive(fcgt_4),282).
orientation(fcgt_1,153,283). coords(fcgt_1,115,126,283).  visible(fcgt_1,283). happensAt(walking(fcgt_1),283). orientation(fcgt_4,166,283). coords(fcgt_4,81,205,283).  visible(fcgt_4,283). happensAt(inactive(fcgt_4),283).
orientation(fcgt_1,153,284). coords(fcgt_1,113,126,284).  visible(fcgt_1,284). happensAt(walking(fcgt_1),284). orientation(fcgt_4,166,284). coords(fcgt_4,82,206,284).  visible(fcgt_4,284). happensAt(inactive(fcgt_4),284).
orientation(fcgt_1,153,285). coords(fcgt_1,112,127,285).  visible(fcgt_1,285). happensAt(walking(fcgt_1),285). orientation(fcgt_4,166,285). coords(fcgt_4,82,206,285).  visible(fcgt_4,285). happensAt(inactive(fcgt_4),285).
orientation(fcgt_1,153,286). coords(fcgt_1,111,127,286).  visible(fcgt_1,286). happensAt(walking(fcgt_1),286). orientation(fcgt_4,166,286). coords(fcgt_4,82,206,286).  visible(fcgt_4,286). happensAt(inactive(fcgt_4),286).
orientation(fcgt_1,153,287). coords(fcgt_1,111,127,287).  visible(fcgt_1,287). happensAt(walking(fcgt_1),287). orientation(fcgt_4,166,287). coords(fcgt_4,82,206,287).  visible(fcgt_4,287). happensAt(inactive(fcgt_4),287).
orientation(fcgt_1,153,288). coords(fcgt_1,110,127,288).  visible(fcgt_1,288). happensAt(walking(fcgt_1),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
orientation(fcgt_1,153,289). coords(fcgt_1,110,127,289).  visible(fcgt_1,289). happensAt(walking(fcgt_1),289). orientation(fcgt_4,166,289). coords(fcgt_4,82,206,289).  visible(fcgt_4,289). happensAt(inactive(fcgt_4),289).
orientation(fcgt_1,153,290). coords(fcgt_1,110,128,290).  visible(fcgt_1,290). happensAt(walking(fcgt_1),290). orientation(fcgt_4,166,290). coords(fcgt_4,82,206,290).  visible(fcgt_4,290). happensAt(inactive(fcgt_4),290).
orientation(fcgt_1,153,291). coords(fcgt_1,109,128,291).  visible(fcgt_1,291). happensAt(walking(fcgt_1),291). orientation(fcgt_4,166,291). coords(fcgt_4,82,206,291).  visible(fcgt_4,291). happensAt(inactive(fcgt_4),291).
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). happensAt(walking(fcgt_1),292). orientation(fcgt_4,166,292). coords(fcgt_4,82,206,292).  visible(fcgt_4,292). happensAt(inactive(fcgt_4),292).
orientation(fcgt_1,153,293). coords(fcgt_1,107,126,293).  visible(fcgt_1,293). happensAt(walking(fcgt_1),293). orientation(fcgt_4,166,293). coords(fcgt_4,82,206,293).  visible(fcgt_4,293). happensAt(inactive(fcgt_4),293).
orientation(fcgt_1,153,294). coords(fcgt_1,104,126,294).  visible(fcgt_1,294). happensAt(walking(fcgt_1),294). orientation(fcgt_4,166,294). coords(fcgt_4,82,206,294).  visible(fcgt_4,294). happensAt(inactive(fcgt_4),294).
orientation(fcgt_1,153,295). coords(fcgt_1,103,127,295).  visible(fcgt_1,295). happensAt(walking(fcgt_1),295). orientation(fcgt_4,166,295). coords(fcgt_4,82,206,295).  visible(fcgt_4,295). happensAt(inactive(fcgt_4),295).
orientation(fcgt_1,153,296). coords(fcgt_1,101,125,296).  visible(fcgt_1,296). happensAt(walking(fcgt_1),296). orientation(fcgt_4,166,296). coords(fcgt_4,82,206,296).  visible(fcgt_4,296). happensAt(inactive(fcgt_4),296).
orientation(fcgt_1,153,297). coords(fcgt_1,101,124,297).  visible(fcgt_1,297). happensAt(active(fcgt_1),297). orientation(fcgt_4,166,297). coords(fcgt_4,82,206,297).  visible(fcgt_4,297). happensAt(inactive(fcgt_4),297).
orientation(fcgt_1,153,298). coords(fcgt_1,101,123,298).  visible(fcgt_1,298). happensAt(active(fcgt_1),298). orientation(fcgt_4,166,298). coords(fcgt_4,82,206,298).  visible(fcgt_4,298). happensAt(inactive(fcgt_4),298).
orientation(fcgt_1,153,299). coords(fcgt_1,100,123,299).  visible(fcgt_1,299). happensAt(active(fcgt_1),299). orientation(fcgt_4,166,299). coords(fcgt_4,82,206,299).  visible(fcgt_4,299). happensAt(inactive(fcgt_4),299).
orientation(fcgt_1,153,300). coords(fcgt_1,100,123,300).  visible(fcgt_1,300). happensAt(active(fcgt_1),300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300). happensAt(inactive(fcgt_4),300).
orientation(fcgt_1,153,301). coords(fcgt_1,99,123,301).  visible(fcgt_1,301). happensAt(active(fcgt_1),301). orientation(fcgt_4,166,301). coords(fcgt_4,82,206,301).  visible(fcgt_4,301). happensAt(inactive(fcgt_4),301).
orientation(fcgt_1,153,302). coords(fcgt_1,99,124,302).  visible(fcgt_1,302). happensAt(active(fcgt_1),302). orientation(fcgt_4,166,302). coords(fcgt_4,82,206,302).  visible(fcgt_4,302). happensAt(inactive(fcgt_4),302).
orientation(fcgt_1,153,303). coords(fcgt_1,98,125,303).  visible(fcgt_1,303). happensAt(active(fcgt_1),303). orientation(fcgt_4,166,303). coords(fcgt_4,82,206,303).  visible(fcgt_4,303). happensAt(inactive(fcgt_4),303).
orientation(fcgt_1,153,304). coords(fcgt_1,98,127,304).  visible(fcgt_1,304). happensAt(active(fcgt_1),304). orientation(fcgt_4,166,304). coords(fcgt_4,82,206,304).  visible(fcgt_4,304). happensAt(inactive(fcgt_4),304).
#show holdsAt/2.