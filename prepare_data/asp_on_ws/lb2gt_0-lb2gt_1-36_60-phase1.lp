
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

	orientation(lb2gt_0,141,36). coords(lb2gt_0,191,125,36).  visible(lb2gt_0,36). happensAt(active(lb2gt_0),36). orientation(lb2gt_1,109,36). coords(lb2gt_1,240,28,36).  visible(lb2gt_1,36). happensAt(walking(lb2gt_1),36).
orientation(lb2gt_0,141,37). coords(lb2gt_0,191,125,37).  visible(lb2gt_0,37). happensAt(active(lb2gt_0),37). orientation(lb2gt_1,107,37). coords(lb2gt_1,241,27,37).  visible(lb2gt_1,37). happensAt(walking(lb2gt_1),37).
orientation(lb2gt_0,141,38). coords(lb2gt_0,191,125,38).  visible(lb2gt_0,38). happensAt(active(lb2gt_0),38). orientation(lb2gt_1,108,38). coords(lb2gt_1,242,25,38).  visible(lb2gt_1,38). happensAt(walking(lb2gt_1),38).
orientation(lb2gt_0,141,39). coords(lb2gt_0,191,125,39).  visible(lb2gt_0,39). happensAt(active(lb2gt_0),39). orientation(lb2gt_1,111,39). coords(lb2gt_1,242,24,39).  visible(lb2gt_1,39). happensAt(walking(lb2gt_1),39).
orientation(lb2gt_0,141,40). coords(lb2gt_0,191,125,40).  visible(lb2gt_0,40). happensAt(active(lb2gt_0),40). orientation(lb2gt_1,107,40). coords(lb2gt_1,242,22,40).  visible(lb2gt_1,40). happensAt(walking(lb2gt_1),40).
orientation(lb2gt_0,141,41). coords(lb2gt_0,191,125,41).  visible(lb2gt_0,41). happensAt(active(lb2gt_0),41). orientation(lb2gt_1,112,41). coords(lb2gt_1,243,20,41).  visible(lb2gt_1,41). happensAt(walking(lb2gt_1),41).
orientation(lb2gt_0,141,42). coords(lb2gt_0,191,125,42).  visible(lb2gt_0,42). happensAt(active(lb2gt_0),42). orientation(lb2gt_1,116,42). coords(lb2gt_1,243,19,42).  visible(lb2gt_1,42). happensAt(walking(lb2gt_1),42).
orientation(lb2gt_0,141,43). coords(lb2gt_0,191,125,43).  visible(lb2gt_0,43). happensAt(active(lb2gt_0),43). orientation(lb2gt_1,116,43). coords(lb2gt_1,245,18,43).  visible(lb2gt_1,43). happensAt(walking(lb2gt_1),43).
orientation(lb2gt_0,141,44). coords(lb2gt_0,191,125,44).  visible(lb2gt_0,44). happensAt(active(lb2gt_0),44). orientation(lb2gt_1,113,44). coords(lb2gt_1,246,18,44).  visible(lb2gt_1,44). happensAt(walking(lb2gt_1),44).
orientation(lb2gt_0,141,45). coords(lb2gt_0,191,125,45).  visible(lb2gt_0,45). happensAt(active(lb2gt_0),45). orientation(lb2gt_1,114,45). coords(lb2gt_1,246,18,45).  visible(lb2gt_1,45). happensAt(walking(lb2gt_1),45).
orientation(lb2gt_0,141,46). coords(lb2gt_0,191,125,46).  visible(lb2gt_0,46). happensAt(active(lb2gt_0),46). orientation(lb2gt_1,117,46). coords(lb2gt_1,247,18,46).  visible(lb2gt_1,46). happensAt(walking(lb2gt_1),46).
orientation(lb2gt_0,141,47). coords(lb2gt_0,191,125,47).  visible(lb2gt_0,47). happensAt(active(lb2gt_0),47). orientation(lb2gt_1,108,47). coords(lb2gt_1,248,18,47).  visible(lb2gt_1,47). happensAt(walking(lb2gt_1),47).
orientation(lb2gt_0,141,48). coords(lb2gt_0,191,125,48).  visible(lb2gt_0,48). happensAt(active(lb2gt_0),48). orientation(lb2gt_1,110,48). coords(lb2gt_1,249,18,48).  visible(lb2gt_1,48). happensAt(walking(lb2gt_1),48).
orientation(lb2gt_0,141,49). coords(lb2gt_0,191,125,49).  visible(lb2gt_0,49). happensAt(active(lb2gt_0),49). orientation(lb2gt_1,104,49). coords(lb2gt_1,250,18,49).  visible(lb2gt_1,49). happensAt(walking(lb2gt_1),49).
orientation(lb2gt_0,141,50). coords(lb2gt_0,191,125,50).  visible(lb2gt_0,50). happensAt(active(lb2gt_0),50). orientation(lb2gt_1,104,50). coords(lb2gt_1,251,18,50).  visible(lb2gt_1,50). happensAt(walking(lb2gt_1),50).
orientation(lb2gt_0,141,51). coords(lb2gt_0,191,125,51).  visible(lb2gt_0,51). happensAt(active(lb2gt_0),51). orientation(lb2gt_1,104,51). coords(lb2gt_1,252,17,51).  visible(lb2gt_1,51). happensAt(walking(lb2gt_1),51).
orientation(lb2gt_0,141,52). coords(lb2gt_0,191,125,52).  visible(lb2gt_0,52). happensAt(active(lb2gt_0),52). orientation(lb2gt_1,104,52). coords(lb2gt_1,252,16,52).  visible(lb2gt_1,52). happensAt(walking(lb2gt_1),52).
orientation(lb2gt_0,141,53). coords(lb2gt_0,191,125,53).  visible(lb2gt_0,53). happensAt(active(lb2gt_0),53). orientation(lb2gt_1,104,53). coords(lb2gt_1,253,15,53).  visible(lb2gt_1,53). happensAt(walking(lb2gt_1),53).
orientation(lb2gt_0,141,54). coords(lb2gt_0,191,125,54).  visible(lb2gt_0,54). happensAt(active(lb2gt_0),54). orientation(lb2gt_1,104,54). coords(lb2gt_1,255,13,54).  visible(lb2gt_1,54). happensAt(walking(lb2gt_1),54).
orientation(lb2gt_0,141,55). coords(lb2gt_0,191,125,55).  visible(lb2gt_0,55). happensAt(active(lb2gt_0),55). orientation(lb2gt_1,104,55). coords(lb2gt_1,255,12,55).  visible(lb2gt_1,55). happensAt(walking(lb2gt_1),55).
orientation(lb2gt_0,141,56). coords(lb2gt_0,191,125,56).  visible(lb2gt_0,56). happensAt(active(lb2gt_0),56). orientation(lb2gt_1,104,56). coords(lb2gt_1,256,10,56).  visible(lb2gt_1,56). happensAt(walking(lb2gt_1),56).
orientation(lb2gt_0,141,57). coords(lb2gt_0,192,125,57).  visible(lb2gt_0,57). happensAt(active(lb2gt_0),57). orientation(lb2gt_1,104,57). coords(lb2gt_1,256,10,57).  visible(lb2gt_1,57). happensAt(walking(lb2gt_1),57).
orientation(lb2gt_0,141,58). coords(lb2gt_0,192,125,58).  visible(lb2gt_0,58). happensAt(active(lb2gt_0),58). orientation(lb2gt_1,104,58). coords(lb2gt_1,256,10,58).  visible(lb2gt_1,58). happensAt(walking(lb2gt_1),58).
orientation(lb2gt_0,141,59). coords(lb2gt_0,193,125,59).  visible(lb2gt_0,59). happensAt(active(lb2gt_0),59). orientation(lb2gt_1,104,59). coords(lb2gt_1,256,10,59).  visible(lb2gt_1,59). happensAt(walking(lb2gt_1),59).
orientation(lb2gt_0,141,60). coords(lb2gt_0,193,124,60).  visible(lb2gt_0,60). happensAt(active(lb2gt_0),60). orientation(lb2gt_1,104,60). coords(lb2gt_1,257,10,60).  visible(lb2gt_1,60). happensAt(walking(lb2gt_1),60).
#show holdsAt/2.