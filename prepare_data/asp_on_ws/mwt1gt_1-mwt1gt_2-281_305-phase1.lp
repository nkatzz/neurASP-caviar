
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

	orientation(mwt1gt_1,160,281). coords(mwt1gt_1,88,174,281).  visible(mwt1gt_1,281). happensAt(walking(mwt1gt_1),281). orientation(mwt1gt_2,130,281). coords(mwt1gt_2,193,67,281).  visible(mwt1gt_2,281). happensAt(walking(mwt1gt_2),281).
orientation(mwt1gt_1,160,282). coords(mwt1gt_1,89,174,282).  visible(mwt1gt_1,282). happensAt(walking(mwt1gt_1),282). orientation(mwt1gt_2,130,282). coords(mwt1gt_2,191,69,282).  visible(mwt1gt_2,282). happensAt(walking(mwt1gt_2),282).
orientation(mwt1gt_1,160,283). coords(mwt1gt_1,89,174,283).  visible(mwt1gt_1,283). happensAt(walking(mwt1gt_1),283). orientation(mwt1gt_2,130,283). coords(mwt1gt_2,191,69,283).  visible(mwt1gt_2,283). happensAt(walking(mwt1gt_2),283).
orientation(mwt1gt_1,160,284). coords(mwt1gt_1,89,173,284).  visible(mwt1gt_1,284). happensAt(walking(mwt1gt_1),284). orientation(mwt1gt_2,130,284). coords(mwt1gt_2,188,70,284).  visible(mwt1gt_2,284). happensAt(walking(mwt1gt_2),284).
orientation(mwt1gt_1,160,285). coords(mwt1gt_1,89,173,285).  visible(mwt1gt_1,285). happensAt(walking(mwt1gt_1),285). orientation(mwt1gt_2,130,285). coords(mwt1gt_2,185,70,285).  visible(mwt1gt_2,285). happensAt(walking(mwt1gt_2),285).
orientation(mwt1gt_1,160,286). coords(mwt1gt_1,90,171,286).  visible(mwt1gt_1,286). happensAt(walking(mwt1gt_1),286). orientation(mwt1gt_2,130,286). coords(mwt1gt_2,183,71,286).  visible(mwt1gt_2,286). happensAt(walking(mwt1gt_2),286).
orientation(mwt1gt_1,160,287). coords(mwt1gt_1,90,171,287).  visible(mwt1gt_1,287). happensAt(walking(mwt1gt_1),287). orientation(mwt1gt_2,130,287). coords(mwt1gt_2,180,72,287).  visible(mwt1gt_2,287). happensAt(walking(mwt1gt_2),287).
orientation(mwt1gt_1,160,288). coords(mwt1gt_1,90,169,288).  visible(mwt1gt_1,288). happensAt(walking(mwt1gt_1),288). orientation(mwt1gt_2,130,288). coords(mwt1gt_2,177,74,288).  visible(mwt1gt_2,288). happensAt(walking(mwt1gt_2),288).
orientation(mwt1gt_1,160,289). coords(mwt1gt_1,91,169,289).  visible(mwt1gt_1,289). happensAt(walking(mwt1gt_1),289). orientation(mwt1gt_2,130,289). coords(mwt1gt_2,176,77,289).  visible(mwt1gt_2,289). happensAt(walking(mwt1gt_2),289).
orientation(mwt1gt_1,160,290). coords(mwt1gt_1,91,168,290).  visible(mwt1gt_1,290). happensAt(walking(mwt1gt_1),290). orientation(mwt1gt_2,130,290). coords(mwt1gt_2,175,79,290).  visible(mwt1gt_2,290). happensAt(walking(mwt1gt_2),290).
orientation(mwt1gt_1,160,291). coords(mwt1gt_1,94,168,291).  visible(mwt1gt_1,291). happensAt(walking(mwt1gt_1),291). orientation(mwt1gt_2,130,291). coords(mwt1gt_2,174,82,291).  visible(mwt1gt_2,291). happensAt(walking(mwt1gt_2),291).
orientation(mwt1gt_1,160,292). coords(mwt1gt_1,96,168,292).  visible(mwt1gt_1,292). happensAt(walking(mwt1gt_1),292). orientation(mwt1gt_2,130,292). coords(mwt1gt_2,173,84,292).  visible(mwt1gt_2,292). happensAt(walking(mwt1gt_2),292).
orientation(mwt1gt_1,160,293). coords(mwt1gt_1,97,167,293).  visible(mwt1gt_1,293). happensAt(walking(mwt1gt_1),293). orientation(mwt1gt_2,130,293). coords(mwt1gt_2,172,86,293).  visible(mwt1gt_2,293). happensAt(walking(mwt1gt_2),293).
orientation(mwt1gt_1,160,294). coords(mwt1gt_1,97,167,294).  visible(mwt1gt_1,294). happensAt(walking(mwt1gt_1),294). orientation(mwt1gt_2,130,294). coords(mwt1gt_2,171,87,294).  visible(mwt1gt_2,294). happensAt(walking(mwt1gt_2),294).
orientation(mwt1gt_1,160,295). coords(mwt1gt_1,98,167,295).  visible(mwt1gt_1,295). happensAt(walking(mwt1gt_1),295). orientation(mwt1gt_2,130,295). coords(mwt1gt_2,171,89,295).  visible(mwt1gt_2,295). happensAt(walking(mwt1gt_2),295).
orientation(mwt1gt_1,160,296). coords(mwt1gt_1,99,167,296).  visible(mwt1gt_1,296). happensAt(walking(mwt1gt_1),296). orientation(mwt1gt_2,130,296). coords(mwt1gt_2,168,89,296).  visible(mwt1gt_2,296). happensAt(walking(mwt1gt_2),296).
orientation(mwt1gt_1,160,297). coords(mwt1gt_1,99,167,297).  visible(mwt1gt_1,297). happensAt(walking(mwt1gt_1),297). orientation(mwt1gt_2,130,297). coords(mwt1gt_2,166,90,297).  visible(mwt1gt_2,297). happensAt(walking(mwt1gt_2),297).
orientation(mwt1gt_1,160,298). coords(mwt1gt_1,100,167,298).  visible(mwt1gt_1,298). happensAt(walking(mwt1gt_1),298). orientation(mwt1gt_2,130,298). coords(mwt1gt_2,164,90,298).  visible(mwt1gt_2,298). happensAt(walking(mwt1gt_2),298).
orientation(mwt1gt_1,160,299). coords(mwt1gt_1,100,167,299).  visible(mwt1gt_1,299). happensAt(walking(mwt1gt_1),299). orientation(mwt1gt_2,130,299). coords(mwt1gt_2,162,92,299).  visible(mwt1gt_2,299). happensAt(walking(mwt1gt_2),299).
orientation(mwt1gt_1,160,300). coords(mwt1gt_1,100,167,300).  visible(mwt1gt_1,300). happensAt(walking(mwt1gt_1),300). orientation(mwt1gt_2,140,300). coords(mwt1gt_2,159,93,300).  visible(mwt1gt_2,300). happensAt(walking(mwt1gt_2),300).
orientation(mwt1gt_1,160,301). coords(mwt1gt_1,100,165,301).  visible(mwt1gt_1,301). happensAt(walking(mwt1gt_1),301). orientation(mwt1gt_2,140,301). coords(mwt1gt_2,158,93,301).  visible(mwt1gt_2,301). happensAt(walking(mwt1gt_2),301).
orientation(mwt1gt_1,160,302). coords(mwt1gt_1,101,165,302).  visible(mwt1gt_1,302). happensAt(walking(mwt1gt_1),302). orientation(mwt1gt_2,140,302). coords(mwt1gt_2,157,94,302).  visible(mwt1gt_2,302). happensAt(walking(mwt1gt_2),302).
orientation(mwt1gt_1,160,303). coords(mwt1gt_1,101,165,303).  visible(mwt1gt_1,303). happensAt(walking(mwt1gt_1),303). orientation(mwt1gt_2,140,303). coords(mwt1gt_2,156,96,303).  visible(mwt1gt_2,303). happensAt(walking(mwt1gt_2),303).
orientation(mwt1gt_1,160,304). coords(mwt1gt_1,101,165,304).  visible(mwt1gt_1,304). happensAt(walking(mwt1gt_1),304). orientation(mwt1gt_2,140,304). coords(mwt1gt_2,156,99,304).  visible(mwt1gt_2,304). happensAt(walking(mwt1gt_2),304).
orientation(mwt1gt_1,160,305). coords(mwt1gt_1,103,165,305).  visible(mwt1gt_1,305). happensAt(walking(mwt1gt_1),305). orientation(mwt1gt_2,140,305). coords(mwt1gt_2,155,99,305).  visible(mwt1gt_2,305). happensAt(walking(mwt1gt_2),305).
#show holdsAt/2.