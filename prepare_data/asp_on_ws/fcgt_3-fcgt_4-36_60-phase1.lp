
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

	orientation(fcgt_3,159,36). coords(fcgt_3,103,163,36).  visible(fcgt_3,36). happensAt(inactive(fcgt_3),36). orientation(fcgt_4,169,36). coords(fcgt_4,101,212,36).  visible(fcgt_4,36). happensAt(walking(fcgt_4),36).
orientation(fcgt_3,159,37). coords(fcgt_3,103,163,37).  visible(fcgt_3,37). happensAt(inactive(fcgt_3),37). orientation(fcgt_4,169,37). coords(fcgt_4,101,212,37).  visible(fcgt_4,37). happensAt(walking(fcgt_4),37).
orientation(fcgt_3,159,38). coords(fcgt_3,103,163,38).  visible(fcgt_3,38). happensAt(inactive(fcgt_3),38). orientation(fcgt_4,169,38). coords(fcgt_4,101,212,38).  visible(fcgt_4,38). happensAt(walking(fcgt_4),38).
orientation(fcgt_3,159,39). coords(fcgt_3,103,163,39).  visible(fcgt_3,39). happensAt(inactive(fcgt_3),39). orientation(fcgt_4,169,39). coords(fcgt_4,101,212,39).  visible(fcgt_4,39). happensAt(walking(fcgt_4),39).
orientation(fcgt_3,159,40). coords(fcgt_3,103,163,40).  visible(fcgt_3,40). happensAt(inactive(fcgt_3),40). orientation(fcgt_4,169,40). coords(fcgt_4,101,212,40).  visible(fcgt_4,40). happensAt(walking(fcgt_4),40).
orientation(fcgt_3,159,41). coords(fcgt_3,103,163,41).  visible(fcgt_3,41). happensAt(inactive(fcgt_3),41). orientation(fcgt_4,169,41). coords(fcgt_4,101,211,41).  visible(fcgt_4,41). happensAt(walking(fcgt_4),41).
orientation(fcgt_3,159,42). coords(fcgt_3,103,163,42).  visible(fcgt_3,42). happensAt(inactive(fcgt_3),42). orientation(fcgt_4,169,42). coords(fcgt_4,101,210,42).  visible(fcgt_4,42). happensAt(walking(fcgt_4),42).
orientation(fcgt_3,159,43). coords(fcgt_3,103,163,43).  visible(fcgt_3,43). happensAt(inactive(fcgt_3),43). orientation(fcgt_4,169,43). coords(fcgt_4,101,210,43).  visible(fcgt_4,43). happensAt(walking(fcgt_4),43).
orientation(fcgt_3,159,44). coords(fcgt_3,103,163,44).  visible(fcgt_3,44). happensAt(inactive(fcgt_3),44). orientation(fcgt_4,169,44). coords(fcgt_4,100,210,44).  visible(fcgt_4,44). happensAt(walking(fcgt_4),44).
orientation(fcgt_3,159,45). coords(fcgt_3,103,163,45).  visible(fcgt_3,45). happensAt(inactive(fcgt_3),45). orientation(fcgt_4,169,45). coords(fcgt_4,100,210,45).  visible(fcgt_4,45). happensAt(walking(fcgt_4),45).
orientation(fcgt_3,159,46). coords(fcgt_3,103,163,46).  visible(fcgt_3,46). happensAt(inactive(fcgt_3),46). orientation(fcgt_4,169,46). coords(fcgt_4,100,210,46).  visible(fcgt_4,46). happensAt(walking(fcgt_4),46).
orientation(fcgt_3,159,47). coords(fcgt_3,103,163,47).  visible(fcgt_3,47). happensAt(inactive(fcgt_3),47). orientation(fcgt_4,169,47). coords(fcgt_4,100,210,47).  visible(fcgt_4,47). happensAt(walking(fcgt_4),47).
orientation(fcgt_3,159,48). coords(fcgt_3,103,163,48).  visible(fcgt_3,48). happensAt(inactive(fcgt_3),48). orientation(fcgt_4,166,48). coords(fcgt_4,100,210,48).  visible(fcgt_4,48). happensAt(walking(fcgt_4),48).
orientation(fcgt_3,159,49). coords(fcgt_3,103,163,49).  visible(fcgt_3,49). happensAt(inactive(fcgt_3),49). orientation(fcgt_4,166,49). coords(fcgt_4,100,210,49).  visible(fcgt_4,49). happensAt(walking(fcgt_4),49).
orientation(fcgt_3,159,50). coords(fcgt_3,103,163,50).  visible(fcgt_3,50). happensAt(inactive(fcgt_3),50). orientation(fcgt_4,165,50). coords(fcgt_4,100,210,50).  visible(fcgt_4,50). happensAt(walking(fcgt_4),50).
orientation(fcgt_3,159,51). coords(fcgt_3,103,163,51).  visible(fcgt_3,51). happensAt(inactive(fcgt_3),51). orientation(fcgt_4,165,51). coords(fcgt_4,100,209,51).  visible(fcgt_4,51). happensAt(inactive(fcgt_4),51).
orientation(fcgt_3,159,52). coords(fcgt_3,103,163,52).  visible(fcgt_3,52). happensAt(inactive(fcgt_3),52). orientation(fcgt_4,165,52). coords(fcgt_4,100,209,52).  visible(fcgt_4,52). happensAt(inactive(fcgt_4),52).
orientation(fcgt_3,159,53). coords(fcgt_3,103,163,53).  visible(fcgt_3,53). happensAt(inactive(fcgt_3),53). orientation(fcgt_4,165,53). coords(fcgt_4,100,209,53).  visible(fcgt_4,53). happensAt(inactive(fcgt_4),53).
orientation(fcgt_3,159,54). coords(fcgt_3,103,163,54).  visible(fcgt_3,54). happensAt(inactive(fcgt_3),54). orientation(fcgt_4,165,54). coords(fcgt_4,100,209,54).  visible(fcgt_4,54). happensAt(inactive(fcgt_4),54).
orientation(fcgt_3,159,55). coords(fcgt_3,103,163,55).  visible(fcgt_3,55). happensAt(inactive(fcgt_3),55). orientation(fcgt_4,165,55). coords(fcgt_4,100,209,55).  visible(fcgt_4,55). happensAt(inactive(fcgt_4),55).
orientation(fcgt_3,159,56). coords(fcgt_3,103,163,56).  visible(fcgt_3,56). happensAt(inactive(fcgt_3),56). orientation(fcgt_4,165,56). coords(fcgt_4,100,209,56).  visible(fcgt_4,56). happensAt(inactive(fcgt_4),56).
orientation(fcgt_3,159,57). coords(fcgt_3,103,163,57).  visible(fcgt_3,57). happensAt(inactive(fcgt_3),57). orientation(fcgt_4,165,57). coords(fcgt_4,100,209,57).  visible(fcgt_4,57). happensAt(inactive(fcgt_4),57).
orientation(fcgt_3,159,58). coords(fcgt_3,103,163,58).  visible(fcgt_3,58). happensAt(inactive(fcgt_3),58). orientation(fcgt_4,165,58). coords(fcgt_4,100,208,58).  visible(fcgt_4,58). happensAt(inactive(fcgt_4),58).
orientation(fcgt_3,159,59). coords(fcgt_3,103,163,59).  visible(fcgt_3,59). happensAt(inactive(fcgt_3),59). orientation(fcgt_4,165,59). coords(fcgt_4,100,208,59).  visible(fcgt_4,59). happensAt(inactive(fcgt_4),59).
orientation(fcgt_3,159,60). coords(fcgt_3,103,163,60).  visible(fcgt_3,60). happensAt(inactive(fcgt_3),60). orientation(fcgt_4,165,60). coords(fcgt_4,100,208,60).  visible(fcgt_4,60). happensAt(inactive(fcgt_4),60).
#show holdsAt/2.