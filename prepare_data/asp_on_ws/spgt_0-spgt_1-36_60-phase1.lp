
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

	orientation(spgt_0,156,36). coords(spgt_0,45,166,36).  visible(spgt_0,36). happensAt(walking(spgt_0),36). orientation(spgt_1,160,36). coords(spgt_1,45,157,36).  visible(spgt_1,36). happensAt(walking(spgt_1),36).
orientation(spgt_0,156,37). coords(spgt_0,45,166,37).  visible(spgt_0,37). happensAt(walking(spgt_0),37). orientation(spgt_1,160,37). coords(spgt_1,45,157,37).  visible(spgt_1,37). happensAt(walking(spgt_1),37).
orientation(spgt_0,156,38). coords(spgt_0,45,166,38).  visible(spgt_0,38). happensAt(walking(spgt_0),38). orientation(spgt_1,160,38). coords(spgt_1,45,157,38).  visible(spgt_1,38). happensAt(walking(spgt_1),38).
orientation(spgt_0,156,39). coords(spgt_0,45,166,39).  visible(spgt_0,39). happensAt(walking(spgt_0),39). orientation(spgt_1,160,39). coords(spgt_1,46,157,39).  visible(spgt_1,39). happensAt(walking(spgt_1),39).
orientation(spgt_0,156,40). coords(spgt_0,45,166,40).  visible(spgt_0,40). happensAt(walking(spgt_0),40). orientation(spgt_1,160,40). coords(spgt_1,46,157,40).  visible(spgt_1,40). happensAt(walking(spgt_1),40).
orientation(spgt_0,156,41). coords(spgt_0,46,166,41).  visible(spgt_0,41). happensAt(walking(spgt_0),41). orientation(spgt_1,160,41). coords(spgt_1,47,157,41).  visible(spgt_1,41). happensAt(walking(spgt_1),41).
orientation(spgt_0,156,42). coords(spgt_0,46,166,42).  visible(spgt_0,42). happensAt(walking(spgt_0),42). orientation(spgt_1,160,42). coords(spgt_1,47,157,42).  visible(spgt_1,42). happensAt(walking(spgt_1),42).
orientation(spgt_0,156,43). coords(spgt_0,46,165,43).  visible(spgt_0,43). happensAt(walking(spgt_0),43). orientation(spgt_1,160,43). coords(spgt_1,47,157,43).  visible(spgt_1,43). happensAt(walking(spgt_1),43).
orientation(spgt_0,156,44). coords(spgt_0,46,165,44).  visible(spgt_0,44). happensAt(walking(spgt_0),44). orientation(spgt_1,160,44). coords(spgt_1,47,156,44).  visible(spgt_1,44). happensAt(walking(spgt_1),44).
orientation(spgt_0,156,45). coords(spgt_0,46,165,45).  visible(spgt_0,45). happensAt(walking(spgt_0),45). orientation(spgt_1,160,45). coords(spgt_1,47,156,45).  visible(spgt_1,45). happensAt(walking(spgt_1),45).
orientation(spgt_0,156,46). coords(spgt_0,46,165,46).  visible(spgt_0,46). happensAt(walking(spgt_0),46). orientation(spgt_1,160,46). coords(spgt_1,48,156,46).  visible(spgt_1,46). happensAt(walking(spgt_1),46).
orientation(spgt_0,156,47). coords(spgt_0,47,164,47).  visible(spgt_0,47). happensAt(walking(spgt_0),47). orientation(spgt_1,160,47). coords(spgt_1,48,156,47).  visible(spgt_1,47). happensAt(walking(spgt_1),47).
orientation(spgt_0,156,48). coords(spgt_0,49,164,48).  visible(spgt_0,48). happensAt(walking(spgt_0),48). orientation(spgt_1,160,48). coords(spgt_1,48,156,48).  visible(spgt_1,48). happensAt(walking(spgt_1),48).
orientation(spgt_0,156,49). coords(spgt_0,49,165,49).  visible(spgt_0,49). happensAt(walking(spgt_0),49). orientation(spgt_1,160,49). coords(spgt_1,49,156,49).  visible(spgt_1,49). happensAt(walking(spgt_1),49).
orientation(spgt_0,156,50). coords(spgt_0,49,165,50).  visible(spgt_0,50). happensAt(walking(spgt_0),50). orientation(spgt_1,160,50). coords(spgt_1,49,156,50).  visible(spgt_1,50). happensAt(walking(spgt_1),50).
orientation(spgt_0,156,51). coords(spgt_0,50,165,51).  visible(spgt_0,51). happensAt(walking(spgt_0),51). orientation(spgt_1,160,51). coords(spgt_1,50,156,51).  visible(spgt_1,51). happensAt(walking(spgt_1),51).
orientation(spgt_0,156,52). coords(spgt_0,50,165,52).  visible(spgt_0,52). happensAt(walking(spgt_0),52). orientation(spgt_1,160,52). coords(spgt_1,51,156,52).  visible(spgt_1,52). happensAt(walking(spgt_1),52).
orientation(spgt_0,156,53). coords(spgt_0,51,164,53).  visible(spgt_0,53). happensAt(walking(spgt_0),53). orientation(spgt_1,160,53). coords(spgt_1,51,156,53).  visible(spgt_1,53). happensAt(walking(spgt_1),53).
orientation(spgt_0,156,54). coords(spgt_0,51,164,54).  visible(spgt_0,54). happensAt(walking(spgt_0),54). orientation(spgt_1,160,54). coords(spgt_1,52,155,54).  visible(spgt_1,54). happensAt(walking(spgt_1),54).
orientation(spgt_0,156,55). coords(spgt_0,52,164,55).  visible(spgt_0,55). happensAt(walking(spgt_0),55). orientation(spgt_1,160,55). coords(spgt_1,52,155,55).  visible(spgt_1,55). happensAt(walking(spgt_1),55).
orientation(spgt_0,156,56). coords(spgt_0,52,164,56).  visible(spgt_0,56). happensAt(walking(spgt_0),56). orientation(spgt_1,160,56). coords(spgt_1,53,155,56).  visible(spgt_1,56). happensAt(walking(spgt_1),56).
orientation(spgt_0,156,57). coords(spgt_0,52,164,57).  visible(spgt_0,57). happensAt(walking(spgt_0),57). orientation(spgt_1,160,57). coords(spgt_1,53,154,57).  visible(spgt_1,57). happensAt(walking(spgt_1),57).
orientation(spgt_0,156,58). coords(spgt_0,52,163,58).  visible(spgt_0,58). happensAt(walking(spgt_0),58). orientation(spgt_1,160,58). coords(spgt_1,53,154,58).  visible(spgt_1,58). happensAt(walking(spgt_1),58).
orientation(spgt_0,156,59). coords(spgt_0,52,163,59).  visible(spgt_0,59). happensAt(walking(spgt_0),59). orientation(spgt_1,160,59). coords(spgt_1,53,153,59).  visible(spgt_1,59). happensAt(walking(spgt_1),59).
orientation(spgt_0,156,60). coords(spgt_0,52,163,60).  visible(spgt_0,60). happensAt(walking(spgt_0),60). orientation(spgt_1,160,60). coords(spgt_1,54,153,60).  visible(spgt_1,60). happensAt(walking(spgt_1),60).
#show holdsAt/2.