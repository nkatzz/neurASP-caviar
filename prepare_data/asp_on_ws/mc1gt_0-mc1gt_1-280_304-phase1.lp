
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

	orientation(mc1gt_0,163,280). coords(mc1gt_0,186,199,280).  visible(mc1gt_0,280). happensAt(walking(mc1gt_0),280). orientation(mc1gt_1,174,280). coords(mc1gt_1,219,211,280).  visible(mc1gt_1,280). happensAt(walking(mc1gt_1),280).
orientation(mc1gt_0,163,281). coords(mc1gt_0,187,202,281).  visible(mc1gt_0,281). happensAt(walking(mc1gt_0),281). orientation(mc1gt_1,178,281). coords(mc1gt_1,220,211,281).  visible(mc1gt_1,281). happensAt(walking(mc1gt_1),281).
orientation(mc1gt_0,163,282). coords(mc1gt_0,189,204,282).  visible(mc1gt_0,282). happensAt(walking(mc1gt_0),282). orientation(mc1gt_1,1,282). coords(mc1gt_1,221,211,282).  visible(mc1gt_1,282). happensAt(walking(mc1gt_1),282).
orientation(mc1gt_0,163,283). coords(mc1gt_0,190,205,283).  visible(mc1gt_0,283). happensAt(walking(mc1gt_0),283). orientation(mc1gt_1,6,283). coords(mc1gt_1,221,212,283).  visible(mc1gt_1,283). happensAt(walking(mc1gt_1),283).
orientation(mc1gt_0,168,284). coords(mc1gt_0,190,207,284).  visible(mc1gt_0,284). happensAt(walking(mc1gt_0),284). orientation(mc1gt_1,4,284). coords(mc1gt_1,222,213,284).  visible(mc1gt_1,284). happensAt(walking(mc1gt_1),284).
orientation(mc1gt_0,168,285). coords(mc1gt_0,189,209,285).  visible(mc1gt_0,285). happensAt(walking(mc1gt_0),285). orientation(mc1gt_1,4,285). coords(mc1gt_1,222,215,285).  visible(mc1gt_1,285). happensAt(walking(mc1gt_1),285).
orientation(mc1gt_0,168,286). coords(mc1gt_0,189,209,286).  visible(mc1gt_0,286). happensAt(walking(mc1gt_0),286). orientation(mc1gt_1,7,286). coords(mc1gt_1,222,218,286).  visible(mc1gt_1,286). happensAt(walking(mc1gt_1),286).
orientation(mc1gt_0,168,287). coords(mc1gt_0,190,212,287).  visible(mc1gt_0,287). happensAt(walking(mc1gt_0),287). orientation(mc1gt_1,3,287). coords(mc1gt_1,222,221,287).  visible(mc1gt_1,287). happensAt(walking(mc1gt_1),287).
orientation(mc1gt_0,168,288). coords(mc1gt_0,191,213,288).  visible(mc1gt_0,288). happensAt(walking(mc1gt_0),288). orientation(mc1gt_1,1,288). coords(mc1gt_1,222,225,288).  visible(mc1gt_1,288). happensAt(walking(mc1gt_1),288).
orientation(mc1gt_0,168,289). coords(mc1gt_0,191,214,289).  visible(mc1gt_0,289). happensAt(walking(mc1gt_0),289). orientation(mc1gt_1,3,289). coords(mc1gt_1,222,228,289).  visible(mc1gt_1,289). happensAt(walking(mc1gt_1),289).
orientation(mc1gt_0,168,290). coords(mc1gt_0,195,216,290).  visible(mc1gt_0,290). happensAt(walking(mc1gt_0),290). orientation(mc1gt_1,0,290). coords(mc1gt_1,222,230,290).  visible(mc1gt_1,290). happensAt(walking(mc1gt_1),290).
orientation(mc1gt_0,174,291). coords(mc1gt_0,196,217,291).  visible(mc1gt_0,291). happensAt(walking(mc1gt_0),291). orientation(mc1gt_1,171,291). coords(mc1gt_1,222,232,291).  visible(mc1gt_1,291). happensAt(walking(mc1gt_1),291).
orientation(mc1gt_0,179,292). coords(mc1gt_0,197,218,292).  visible(mc1gt_0,292). happensAt(walking(mc1gt_0),292). orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292).
orientation(mc1gt_0,1,293). coords(mc1gt_0,197,219,293).  visible(mc1gt_0,293). happensAt(walking(mc1gt_0),293). orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293).
orientation(mc1gt_0,1,294). coords(mc1gt_0,198,220,294).  visible(mc1gt_0,294). happensAt(walking(mc1gt_0),294). orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294).
orientation(mc1gt_0,1,295). coords(mc1gt_0,199,221,295).  visible(mc1gt_0,295). happensAt(walking(mc1gt_0),295). orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295).
orientation(mc1gt_0,1,296). coords(mc1gt_0,199,223,296).  visible(mc1gt_0,296). happensAt(walking(mc1gt_0),296). orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296).
orientation(mc1gt_0,1,297). coords(mc1gt_0,199,224,297).  visible(mc1gt_0,297). happensAt(walking(mc1gt_0),297). orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297).
orientation(mc1gt_0,1,298). coords(mc1gt_0,199,225,298).  visible(mc1gt_0,298). happensAt(walking(mc1gt_0),298). orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298).
orientation(mc1gt_0,1,299). coords(mc1gt_0,199,229,299).  visible(mc1gt_0,299). happensAt(walking(mc1gt_0),299). orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299).
orientation(mc1gt_0,1,300). coords(mc1gt_0,199,231,300).  visible(mc1gt_0,300). happensAt(walking(mc1gt_0),300). orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300).
orientation(mc1gt_0,1,301). coords(mc1gt_0,200,231,301).  visible(mc1gt_0,301). happensAt(walking(mc1gt_0),301). orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301).
orientation(mc1gt_0,1,302). coords(mc1gt_0,200,232,302).  visible(mc1gt_0,302). happensAt(walking(mc1gt_0),302). orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302).
orientation(mc1gt_0,1,303). coords(mc1gt_0,201,233,303).  visible(mc1gt_0,303). happensAt(walking(mc1gt_0),303). orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303).
orientation(mc1gt_0,1,304). coords(mc1gt_0,202,233,304).  visible(mc1gt_0,304). happensAt(walking(mc1gt_0),304). orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304).
#show holdsAt/2.