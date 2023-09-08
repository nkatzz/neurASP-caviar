
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

	orientation(fra1gt_6,128,282). coords(fra1gt_6,202,70,282).  visible(fra1gt_6,282). happensAt(walking(fra1gt_6),282). orientation(fra1gt_7,137,282). coords(fra1gt_7,140,47,282).  visible(fra1gt_7,282). happensAt(walking(fra1gt_7),282).
orientation(fra1gt_6,128,283). coords(fra1gt_6,202,71,283).  visible(fra1gt_6,283). happensAt(walking(fra1gt_6),283). orientation(fra1gt_7,137,283). coords(fra1gt_7,140,47,283).  visible(fra1gt_7,283). happensAt(walking(fra1gt_7),283).
orientation(fra1gt_6,128,284). coords(fra1gt_6,201,71,284).  visible(fra1gt_6,284). happensAt(walking(fra1gt_6),284). orientation(fra1gt_7,137,284). coords(fra1gt_7,139,47,284).  visible(fra1gt_7,284). happensAt(walking(fra1gt_7),284).
orientation(fra1gt_6,128,285). coords(fra1gt_6,200,72,285).  visible(fra1gt_6,285). happensAt(walking(fra1gt_6),285). orientation(fra1gt_7,142,285). coords(fra1gt_7,139,47,285).  visible(fra1gt_7,285). happensAt(walking(fra1gt_7),285).
orientation(fra1gt_6,128,286). coords(fra1gt_6,199,72,286).  visible(fra1gt_6,286). happensAt(walking(fra1gt_6),286). orientation(fra1gt_7,142,286). coords(fra1gt_7,137,47,286).  visible(fra1gt_7,286). happensAt(walking(fra1gt_7),286).
orientation(fra1gt_6,128,287). coords(fra1gt_6,198,72,287).  visible(fra1gt_6,287). happensAt(walking(fra1gt_6),287). orientation(fra1gt_7,142,287). coords(fra1gt_7,137,48,287).  visible(fra1gt_7,287). happensAt(walking(fra1gt_7),287).
orientation(fra1gt_6,128,288). coords(fra1gt_6,197,73,288).  visible(fra1gt_6,288). happensAt(walking(fra1gt_6),288). orientation(fra1gt_7,142,288). coords(fra1gt_7,137,49,288).  visible(fra1gt_7,288). happensAt(walking(fra1gt_7),288).
orientation(fra1gt_6,128,289). coords(fra1gt_6,195,73,289).  visible(fra1gt_6,289). happensAt(walking(fra1gt_6),289). orientation(fra1gt_7,142,289). coords(fra1gt_7,137,50,289).  visible(fra1gt_7,289). happensAt(walking(fra1gt_7),289).
orientation(fra1gt_6,128,290). coords(fra1gt_6,194,74,290).  visible(fra1gt_6,290). happensAt(walking(fra1gt_6),290). orientation(fra1gt_7,142,290). coords(fra1gt_7,137,51,290).  visible(fra1gt_7,290). happensAt(walking(fra1gt_7),290).
orientation(fra1gt_6,128,291). coords(fra1gt_6,191,74,291).  visible(fra1gt_6,291). happensAt(walking(fra1gt_6),291). orientation(fra1gt_7,142,291). coords(fra1gt_7,137,51,291).  visible(fra1gt_7,291). happensAt(walking(fra1gt_7),291).
orientation(fra1gt_6,128,292). coords(fra1gt_6,190,74,292).  visible(fra1gt_6,292). happensAt(walking(fra1gt_6),292). orientation(fra1gt_7,142,292). coords(fra1gt_7,137,51,292).  visible(fra1gt_7,292). happensAt(walking(fra1gt_7),292).
orientation(fra1gt_6,128,293). coords(fra1gt_6,188,74,293).  visible(fra1gt_6,293). happensAt(walking(fra1gt_6),293). orientation(fra1gt_7,142,293). coords(fra1gt_7,137,51,293).  visible(fra1gt_7,293). happensAt(walking(fra1gt_7),293).
orientation(fra1gt_6,128,294). coords(fra1gt_6,185,75,294).  visible(fra1gt_6,294). happensAt(walking(fra1gt_6),294). orientation(fra1gt_7,142,294). coords(fra1gt_7,137,51,294).  visible(fra1gt_7,294). happensAt(walking(fra1gt_7),294).
orientation(fra1gt_6,128,295). coords(fra1gt_6,185,75,295).  visible(fra1gt_6,295). happensAt(walking(fra1gt_6),295). orientation(fra1gt_7,142,295). coords(fra1gt_7,137,52,295).  visible(fra1gt_7,295). happensAt(walking(fra1gt_7),295).
orientation(fra1gt_6,128,296). coords(fra1gt_6,184,75,296).  visible(fra1gt_6,296). happensAt(walking(fra1gt_6),296). orientation(fra1gt_7,142,296). coords(fra1gt_7,137,52,296).  visible(fra1gt_7,296). happensAt(walking(fra1gt_7),296).
orientation(fra1gt_6,128,297). coords(fra1gt_6,183,75,297).  visible(fra1gt_6,297). happensAt(walking(fra1gt_6),297). orientation(fra1gt_7,142,297). coords(fra1gt_7,137,52,297).  visible(fra1gt_7,297). happensAt(walking(fra1gt_7),297).
orientation(fra1gt_6,128,298). coords(fra1gt_6,182,76,298).  visible(fra1gt_6,298). happensAt(walking(fra1gt_6),298). orientation(fra1gt_7,142,298). coords(fra1gt_7,137,52,298).  visible(fra1gt_7,298). happensAt(walking(fra1gt_7),298).
orientation(fra1gt_6,128,299). coords(fra1gt_6,182,76,299).  visible(fra1gt_6,299). happensAt(walking(fra1gt_6),299). orientation(fra1gt_7,142,299). coords(fra1gt_7,137,53,299).  visible(fra1gt_7,299). happensAt(walking(fra1gt_7),299).
orientation(fra1gt_6,128,300). coords(fra1gt_6,182,76,300).  visible(fra1gt_6,300). happensAt(walking(fra1gt_6),300). orientation(fra1gt_7,142,300). coords(fra1gt_7,137,55,300).  visible(fra1gt_7,300). happensAt(walking(fra1gt_7),300).
orientation(fra1gt_6,128,301). coords(fra1gt_6,181,76,301).  visible(fra1gt_6,301). happensAt(walking(fra1gt_6),301). orientation(fra1gt_7,142,301). coords(fra1gt_7,138,57,301).  visible(fra1gt_7,301). happensAt(walking(fra1gt_7),301).
orientation(fra1gt_6,131,302). coords(fra1gt_6,181,76,302).  visible(fra1gt_6,302). happensAt(walking(fra1gt_6),302). orientation(fra1gt_7,142,302). coords(fra1gt_7,137,59,302).  visible(fra1gt_7,302). happensAt(walking(fra1gt_7),302).
orientation(fra1gt_6,133,303). coords(fra1gt_6,180,76,303).  visible(fra1gt_6,303). happensAt(walking(fra1gt_6),303). orientation(fra1gt_7,142,303). coords(fra1gt_7,138,59,303).  visible(fra1gt_7,303). happensAt(walking(fra1gt_7),303).
orientation(fra1gt_6,133,304). coords(fra1gt_6,180,76,304).  visible(fra1gt_6,304). happensAt(walking(fra1gt_6),304). orientation(fra1gt_7,142,304). coords(fra1gt_7,138,60,304).  visible(fra1gt_7,304). happensAt(walking(fra1gt_7),304).
orientation(fra1gt_6,133,305). coords(fra1gt_6,177,76,305).  visible(fra1gt_6,305). happensAt(walking(fra1gt_6),305). orientation(fra1gt_7,142,305). coords(fra1gt_7,138,62,305).  visible(fra1gt_7,305). happensAt(walking(fra1gt_7),305).
orientation(fra1gt_6,133,306). coords(fra1gt_6,175,76,306).  visible(fra1gt_6,306). happensAt(walking(fra1gt_6),306). orientation(fra1gt_7,142,306). coords(fra1gt_7,138,62,306).  visible(fra1gt_7,306). happensAt(walking(fra1gt_7),306).
#show holdsAt/2.