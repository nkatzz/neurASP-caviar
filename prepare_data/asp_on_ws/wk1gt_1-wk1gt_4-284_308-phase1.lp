
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

	orientation(wk1gt_1,155,284). coords(wk1gt_1,203,218,284).  visible(wk1gt_1,284). happensAt(walking(wk1gt_1),284). orientation(wk1gt_4,159,284). coords(wk1gt_4,51,176,284).  visible(wk1gt_4,284). happensAt(active(wk1gt_4),284).
orientation(wk1gt_1,165,285). coords(wk1gt_1,202,216,285).  visible(wk1gt_1,285). happensAt(walking(wk1gt_1),285). orientation(wk1gt_4,160,285). coords(wk1gt_4,51,176,285).  visible(wk1gt_4,285). happensAt(active(wk1gt_4),285).
orientation(wk1gt_1,165,286). coords(wk1gt_1,202,214,286).  visible(wk1gt_1,286). happensAt(walking(wk1gt_1),286). orientation(wk1gt_4,160,286). coords(wk1gt_4,51,177,286).  visible(wk1gt_4,286). happensAt(active(wk1gt_4),286).
orientation(wk1gt_1,165,287). coords(wk1gt_1,201,212,287).  visible(wk1gt_1,287). happensAt(walking(wk1gt_1),287). orientation(wk1gt_4,160,287). coords(wk1gt_4,51,177,287).  visible(wk1gt_4,287). happensAt(active(wk1gt_4),287).
orientation(wk1gt_1,165,288). coords(wk1gt_1,201,210,288).  visible(wk1gt_1,288). happensAt(walking(wk1gt_1),288). orientation(wk1gt_4,158,288). coords(wk1gt_4,52,177,288).  visible(wk1gt_4,288). happensAt(active(wk1gt_4),288).
orientation(wk1gt_1,165,289). coords(wk1gt_1,200,209,289).  visible(wk1gt_1,289). happensAt(walking(wk1gt_1),289). orientation(wk1gt_4,166,289). coords(wk1gt_4,52,177,289).  visible(wk1gt_4,289). happensAt(active(wk1gt_4),289).
orientation(wk1gt_1,160,290). coords(wk1gt_1,200,208,290).  visible(wk1gt_1,290). happensAt(walking(wk1gt_1),290). orientation(wk1gt_4,160,290). coords(wk1gt_4,52,177,290).  visible(wk1gt_4,290). happensAt(active(wk1gt_4),290).
orientation(wk1gt_1,154,291). coords(wk1gt_1,200,207,291).  visible(wk1gt_1,291). happensAt(walking(wk1gt_1),291). orientation(wk1gt_4,157,291). coords(wk1gt_4,52,177,291).  visible(wk1gt_4,291). happensAt(active(wk1gt_4),291).
orientation(wk1gt_1,154,292). coords(wk1gt_1,198,206,292).  visible(wk1gt_1,292). happensAt(walking(wk1gt_1),292). orientation(wk1gt_4,162,292). coords(wk1gt_4,52,177,292).  visible(wk1gt_4,292). happensAt(active(wk1gt_4),292).
orientation(wk1gt_1,154,293). coords(wk1gt_1,198,204,293).  visible(wk1gt_1,293). happensAt(walking(wk1gt_1),293). orientation(wk1gt_4,161,293). coords(wk1gt_4,52,177,293).  visible(wk1gt_4,293). happensAt(active(wk1gt_4),293).
orientation(wk1gt_1,154,294). coords(wk1gt_1,196,202,294).  visible(wk1gt_1,294). happensAt(walking(wk1gt_1),294). orientation(wk1gt_4,157,294). coords(wk1gt_4,52,177,294).  visible(wk1gt_4,294). happensAt(active(wk1gt_4),294).
orientation(wk1gt_1,163,295). coords(wk1gt_1,193,200,295).  visible(wk1gt_1,295). happensAt(walking(wk1gt_1),295). orientation(wk1gt_4,158,295). coords(wk1gt_4,52,177,295).  visible(wk1gt_4,295). happensAt(active(wk1gt_4),295).
orientation(wk1gt_1,163,296). coords(wk1gt_1,193,198,296).  visible(wk1gt_1,296). happensAt(walking(wk1gt_1),296). orientation(wk1gt_4,161,296). coords(wk1gt_4,52,176,296).  visible(wk1gt_4,296). happensAt(active(wk1gt_4),296).
orientation(wk1gt_1,163,297). coords(wk1gt_1,191,193,297).  visible(wk1gt_1,297). happensAt(walking(wk1gt_1),297). orientation(wk1gt_4,160,297). coords(wk1gt_4,52,176,297).  visible(wk1gt_4,297). happensAt(active(wk1gt_4),297).
orientation(wk1gt_1,163,298). coords(wk1gt_1,191,192,298).  visible(wk1gt_1,298). happensAt(walking(wk1gt_1),298). orientation(wk1gt_4,161,298). coords(wk1gt_4,52,175,298).  visible(wk1gt_4,298). happensAt(active(wk1gt_4),298).
orientation(wk1gt_1,163,299). coords(wk1gt_1,189,191,299).  visible(wk1gt_1,299). happensAt(walking(wk1gt_1),299). orientation(wk1gt_4,167,299). coords(wk1gt_4,52,175,299).  visible(wk1gt_4,299). happensAt(active(wk1gt_4),299).
orientation(wk1gt_1,155,300). coords(wk1gt_1,189,191,300).  visible(wk1gt_1,300). happensAt(walking(wk1gt_1),300). orientation(wk1gt_4,168,300). coords(wk1gt_4,52,174,300).  visible(wk1gt_4,300). happensAt(active(wk1gt_4),300).
orientation(wk1gt_1,155,301). coords(wk1gt_1,188,190,301).  visible(wk1gt_1,301). happensAt(walking(wk1gt_1),301). orientation(wk1gt_4,173,301). coords(wk1gt_4,52,174,301).  visible(wk1gt_4,301). happensAt(active(wk1gt_4),301).
orientation(wk1gt_1,155,302). coords(wk1gt_1,187,189,302).  visible(wk1gt_1,302). happensAt(walking(wk1gt_1),302). orientation(wk1gt_4,170,302). coords(wk1gt_4,52,173,302).  visible(wk1gt_4,302). happensAt(active(wk1gt_4),302).
orientation(wk1gt_1,155,303). coords(wk1gt_1,187,188,303).  visible(wk1gt_1,303). happensAt(walking(wk1gt_1),303). orientation(wk1gt_4,170,303). coords(wk1gt_4,52,173,303).  visible(wk1gt_4,303). happensAt(active(wk1gt_4),303).
orientation(wk1gt_1,155,304). coords(wk1gt_1,186,187,304).  visible(wk1gt_1,304). happensAt(walking(wk1gt_1),304). orientation(wk1gt_4,170,304). coords(wk1gt_4,52,173,304).  visible(wk1gt_4,304). happensAt(active(wk1gt_4),304).
orientation(wk1gt_1,155,305). coords(wk1gt_1,185,186,305).  visible(wk1gt_1,305). happensAt(walking(wk1gt_1),305). orientation(wk1gt_4,169,305). coords(wk1gt_4,52,173,305).  visible(wk1gt_4,305). happensAt(active(wk1gt_4),305).
orientation(wk1gt_1,155,306). coords(wk1gt_1,183,184,306).  visible(wk1gt_1,306). happensAt(walking(wk1gt_1),306). orientation(wk1gt_4,168,306). coords(wk1gt_4,52,173,306).  visible(wk1gt_4,306). happensAt(active(wk1gt_4),306).
orientation(wk1gt_1,155,307). coords(wk1gt_1,182,181,307).  visible(wk1gt_1,307). happensAt(walking(wk1gt_1),307). orientation(wk1gt_4,167,307). coords(wk1gt_4,52,173,307).  visible(wk1gt_4,307). happensAt(active(wk1gt_4),307).
orientation(wk1gt_1,142,308). coords(wk1gt_1,179,179,308).  visible(wk1gt_1,308). happensAt(walking(wk1gt_1),308). orientation(wk1gt_4,167,308). coords(wk1gt_4,52,173,308).  visible(wk1gt_4,308). happensAt(active(wk1gt_4),308).
#show holdsAt/2.