
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

	orientation(mc1gt_0,152,268). coords(mc1gt_0,182,179,268).  visible(mc1gt_0,268). happensAt(walking(mc1gt_0),268). orientation(mc1gt_1,175,268). coords(mc1gt_1,209,187,268).  visible(mc1gt_1,268). happensAt(walking(mc1gt_1),268).
orientation(mc1gt_0,152,269). coords(mc1gt_0,182,182,269).  visible(mc1gt_0,269). happensAt(walking(mc1gt_0),269). orientation(mc1gt_1,172,269). coords(mc1gt_1,209,189,269).  visible(mc1gt_1,269). happensAt(walking(mc1gt_1),269).
orientation(mc1gt_0,152,270). coords(mc1gt_0,183,185,270).  visible(mc1gt_0,270). happensAt(walking(mc1gt_0),270). orientation(mc1gt_1,177,270). coords(mc1gt_1,210,190,270).  visible(mc1gt_1,270). happensAt(walking(mc1gt_1),270).
orientation(mc1gt_0,152,271). coords(mc1gt_0,183,188,271).  visible(mc1gt_0,271). happensAt(walking(mc1gt_0),271). orientation(mc1gt_1,177,271). coords(mc1gt_1,211,192,271).  visible(mc1gt_1,271). happensAt(walking(mc1gt_1),271).
orientation(mc1gt_0,156,272). coords(mc1gt_0,184,191,272).  visible(mc1gt_0,272). happensAt(walking(mc1gt_0),272). orientation(mc1gt_1,163,272). coords(mc1gt_1,212,193,272).  visible(mc1gt_1,272). happensAt(walking(mc1gt_1),272).
orientation(mc1gt_0,156,273). coords(mc1gt_0,184,192,273).  visible(mc1gt_0,273). happensAt(walking(mc1gt_0),273). orientation(mc1gt_1,165,273). coords(mc1gt_1,213,194,273).  visible(mc1gt_1,273). happensAt(walking(mc1gt_1),273).
orientation(mc1gt_0,156,274). coords(mc1gt_0,185,194,274).  visible(mc1gt_0,274). happensAt(walking(mc1gt_0),274). orientation(mc1gt_1,168,274). coords(mc1gt_1,214,196,274).  visible(mc1gt_1,274). happensAt(walking(mc1gt_1),274).
orientation(mc1gt_0,156,275). coords(mc1gt_0,186,195,275).  visible(mc1gt_0,275). happensAt(walking(mc1gt_0),275). orientation(mc1gt_1,171,275). coords(mc1gt_1,215,200,275).  visible(mc1gt_1,275). happensAt(walking(mc1gt_1),275).
orientation(mc1gt_0,156,276). coords(mc1gt_0,186,195,276).  visible(mc1gt_0,276). happensAt(walking(mc1gt_0),276). orientation(mc1gt_1,170,276). coords(mc1gt_1,217,204,276).  visible(mc1gt_1,276). happensAt(walking(mc1gt_1),276).
orientation(mc1gt_0,156,277). coords(mc1gt_0,186,196,277).  visible(mc1gt_0,277). happensAt(walking(mc1gt_0),277). orientation(mc1gt_1,169,277). coords(mc1gt_1,218,208,277).  visible(mc1gt_1,277). happensAt(walking(mc1gt_1),277).
orientation(mc1gt_0,163,278). coords(mc1gt_0,186,196,278).  visible(mc1gt_0,278). happensAt(walking(mc1gt_0),278). orientation(mc1gt_1,175,278). coords(mc1gt_1,218,210,278).  visible(mc1gt_1,278). happensAt(walking(mc1gt_1),278).
orientation(mc1gt_0,163,279). coords(mc1gt_0,186,198,279).  visible(mc1gt_0,279). happensAt(walking(mc1gt_0),279). orientation(mc1gt_1,175,279). coords(mc1gt_1,219,211,279).  visible(mc1gt_1,279). happensAt(walking(mc1gt_1),279).
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
#show holdsAt/2.