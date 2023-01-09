
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

	orientation(wk1gt_4,145,267). coords(wk1gt_4,48,179,267).  visible(wk1gt_4,267). happensAt(active(wk1gt_4),267). orientation(wk1gt_5,162,267). coords(wk1gt_5,50,166,267).  visible(wk1gt_5,267). happensAt(inactive(wk1gt_5),267).
orientation(wk1gt_4,152,268). coords(wk1gt_4,48,179,268).  visible(wk1gt_4,268). happensAt(active(wk1gt_4),268). orientation(wk1gt_5,162,268). coords(wk1gt_5,50,166,268).  visible(wk1gt_5,268). happensAt(inactive(wk1gt_5),268).
orientation(wk1gt_4,156,269). coords(wk1gt_4,48,178,269).  visible(wk1gt_4,269). happensAt(active(wk1gt_4),269). orientation(wk1gt_5,162,269). coords(wk1gt_5,50,166,269).  visible(wk1gt_5,269). happensAt(inactive(wk1gt_5),269).
orientation(wk1gt_4,158,270). coords(wk1gt_4,48,178,270).  visible(wk1gt_4,270). happensAt(active(wk1gt_4),270). orientation(wk1gt_5,162,270). coords(wk1gt_5,50,166,270).  visible(wk1gt_5,270). happensAt(inactive(wk1gt_5),270).
orientation(wk1gt_4,157,271). coords(wk1gt_4,49,178,271).  visible(wk1gt_4,271). happensAt(active(wk1gt_4),271). orientation(wk1gt_5,162,271). coords(wk1gt_5,50,166,271).  visible(wk1gt_5,271). happensAt(inactive(wk1gt_5),271).
orientation(wk1gt_4,153,272). coords(wk1gt_4,49,178,272).  visible(wk1gt_4,272). happensAt(active(wk1gt_4),272). orientation(wk1gt_5,162,272). coords(wk1gt_5,50,166,272).  visible(wk1gt_5,272). happensAt(active(wk1gt_5),272).
orientation(wk1gt_4,157,273). coords(wk1gt_4,49,178,273).  visible(wk1gt_4,273). happensAt(active(wk1gt_4),273). orientation(wk1gt_5,162,273). coords(wk1gt_5,50,166,273).  visible(wk1gt_5,273). happensAt(active(wk1gt_5),273).
orientation(wk1gt_4,156,274). coords(wk1gt_4,49,178,274).  visible(wk1gt_4,274). happensAt(active(wk1gt_4),274). orientation(wk1gt_5,162,274). coords(wk1gt_5,50,166,274).  visible(wk1gt_5,274). happensAt(active(wk1gt_5),274).
orientation(wk1gt_4,153,275). coords(wk1gt_4,49,178,275).  visible(wk1gt_4,275). happensAt(active(wk1gt_4),275). orientation(wk1gt_5,162,275). coords(wk1gt_5,50,166,275).  visible(wk1gt_5,275). happensAt(active(wk1gt_5),275).
orientation(wk1gt_4,154,276). coords(wk1gt_4,49,177,276).  visible(wk1gt_4,276). happensAt(active(wk1gt_4),276). orientation(wk1gt_5,162,276). coords(wk1gt_5,50,166,276).  visible(wk1gt_5,276). happensAt(active(wk1gt_5),276).
orientation(wk1gt_4,154,277). coords(wk1gt_4,49,177,277).  visible(wk1gt_4,277). happensAt(active(wk1gt_4),277). orientation(wk1gt_5,162,277). coords(wk1gt_5,50,166,277).  visible(wk1gt_5,277). happensAt(active(wk1gt_5),277).
orientation(wk1gt_4,156,278). coords(wk1gt_4,50,177,278).  visible(wk1gt_4,278). happensAt(active(wk1gt_4),278). orientation(wk1gt_5,162,278). coords(wk1gt_5,50,166,278).  visible(wk1gt_5,278). happensAt(active(wk1gt_5),278).
orientation(wk1gt_4,155,279). coords(wk1gt_4,51,177,279).  visible(wk1gt_4,279). happensAt(active(wk1gt_4),279). orientation(wk1gt_5,162,279). coords(wk1gt_5,50,166,279).  visible(wk1gt_5,279). happensAt(active(wk1gt_5),279).
orientation(wk1gt_4,156,280). coords(wk1gt_4,51,177,280).  visible(wk1gt_4,280). happensAt(active(wk1gt_4),280). orientation(wk1gt_5,162,280). coords(wk1gt_5,50,166,280).  visible(wk1gt_5,280). happensAt(active(wk1gt_5),280).
orientation(wk1gt_4,157,281). coords(wk1gt_4,51,176,281).  visible(wk1gt_4,281). happensAt(active(wk1gt_4),281). orientation(wk1gt_5,162,281). coords(wk1gt_5,50,166,281).  visible(wk1gt_5,281). happensAt(active(wk1gt_5),281).
orientation(wk1gt_4,161,282). coords(wk1gt_4,51,176,282).  visible(wk1gt_4,282). happensAt(active(wk1gt_4),282). orientation(wk1gt_5,162,282). coords(wk1gt_5,50,166,282).  visible(wk1gt_5,282). happensAt(active(wk1gt_5),282).
orientation(wk1gt_4,162,283). coords(wk1gt_4,51,176,283).  visible(wk1gt_4,283). happensAt(active(wk1gt_4),283). orientation(wk1gt_5,162,283). coords(wk1gt_5,50,166,283).  visible(wk1gt_5,283). happensAt(active(wk1gt_5),283).
orientation(wk1gt_4,159,284). coords(wk1gt_4,51,176,284).  visible(wk1gt_4,284). happensAt(active(wk1gt_4),284). orientation(wk1gt_5,162,284). coords(wk1gt_5,50,166,284).  visible(wk1gt_5,284). happensAt(active(wk1gt_5),284).
orientation(wk1gt_4,160,285). coords(wk1gt_4,51,176,285).  visible(wk1gt_4,285). happensAt(active(wk1gt_4),285). orientation(wk1gt_5,162,285). coords(wk1gt_5,50,166,285).  visible(wk1gt_5,285). happensAt(active(wk1gt_5),285).
orientation(wk1gt_4,160,286). coords(wk1gt_4,51,177,286).  visible(wk1gt_4,286). happensAt(active(wk1gt_4),286). orientation(wk1gt_5,162,286). coords(wk1gt_5,50,166,286).  visible(wk1gt_5,286). happensAt(active(wk1gt_5),286).
orientation(wk1gt_4,160,287). coords(wk1gt_4,51,177,287).  visible(wk1gt_4,287). happensAt(active(wk1gt_4),287). orientation(wk1gt_5,162,287). coords(wk1gt_5,50,166,287).  visible(wk1gt_5,287). happensAt(active(wk1gt_5),287).
orientation(wk1gt_4,158,288). coords(wk1gt_4,52,177,288).  visible(wk1gt_4,288). happensAt(active(wk1gt_4),288). orientation(wk1gt_5,162,288). coords(wk1gt_5,50,166,288).  visible(wk1gt_5,288). happensAt(active(wk1gt_5),288).
orientation(wk1gt_4,166,289). coords(wk1gt_4,52,177,289).  visible(wk1gt_4,289). happensAt(active(wk1gt_4),289). orientation(wk1gt_5,162,289). coords(wk1gt_5,50,166,289).  visible(wk1gt_5,289). happensAt(active(wk1gt_5),289).
orientation(wk1gt_4,160,290). coords(wk1gt_4,52,177,290).  visible(wk1gt_4,290). happensAt(active(wk1gt_4),290). orientation(wk1gt_5,162,290). coords(wk1gt_5,50,165,290).  visible(wk1gt_5,290). happensAt(active(wk1gt_5),290).
orientation(wk1gt_4,157,291). coords(wk1gt_4,52,177,291).  visible(wk1gt_4,291). happensAt(active(wk1gt_4),291). orientation(wk1gt_5,162,291). coords(wk1gt_5,50,165,291).  visible(wk1gt_5,291). happensAt(active(wk1gt_5),291).
#show holdsAt/2.